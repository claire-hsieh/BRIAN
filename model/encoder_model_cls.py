import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
import torch.optim as optim
import torch.nn.init as init
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold, GroupKFold, LeaveOneGroupOut
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import spearmanr
import random
import os
import regex as re
import h5py
from tqdm import tqdm
import psutil
import time
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from collections import Counter

def load_output(input_file, protein_labels = False, order = False):
    with h5py.File(input_file, 'r') as f:
        x = torch.tensor(f['x'][:], dtype=torch.float32)
        y = torch.tensor(f['y'][:], dtype=torch.float32)
        mask = torch.tensor(f['mask'][:], dtype=torch.float32)
        print(f"Loaded data from {input_file}")
        if protein_labels:
            group = f['protein_labels']
            protein_labels = [group[f'sublist_{i}'][:] for i in range(len(group))]
            return x, y, protein_labels, mask
        if order:
            order_labels = f['order_labels'][:]
            if isinstance(order_labels[0], bytes):
                order_labels = [label.decode('utf-8') for label in order_labels]
            # convert to int
            vocab = {label: i for i, label in enumerate(set(order_labels))}
            order_labels = [vocab[label] for label in order_labels]
            return x, y, order_labels, vocab, mask
        return x, y, mask

class ESM2_Concat(Dataset):
    def __init__(self, x, y, mask):# , species_id):
        self.x = x 
        self.y = y 
        self.mask = mask
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], mask[idx]
    

def create_dataloader(data, labels, pad_mask, batch_size, is_train):
    dataset = ESM2_Concat(data, labels, pad_mask)
    if is_train:
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    else:
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)
# Seeding everything for reproducibility
def set_seed(seed=None):
    if seed is None:
       seed = np.random.choice(int(1e2))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    seed_everything(seed)


class TransformerRegressor(pl.LightningModule):
    def __init__(self, input_dim, seq_len, num_heads, hidden_dim, num_layers, lr=1e-3):
        super().__init__()
        self.input_dim = input_dim
        self.cls_token = nn.Parameter(torch.zeros(1, 1, input_dim).to(torch.float32))  # Learnable CLS token
        # self.positional_encoding = PositionalEncoding(input_dim,seq_len + 1)  # + 1 for CLS token
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, dim_feedforward=hidden_dim, batch_first=True)
        # define your own attention layer 
        # swap out with linformer

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.regression_head = nn.Linear(input_dim, 1)
        self.lr = lr
        self.criterion = nn.MSELoss()

        self.train_losses = []
        self.val_losses = []

        self._initialize_weights()

    def _initialize_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                if len(param.shape) > 1:
                    init.kaiming_uniform_(param, nonlinearity='relu')
            elif 'bias' in name:
                init.constant_(param, 0)

    def forward(self, x, mask):
        batch_size = x.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # CLS token expanded for every sequence in batch
        x = torch.cat((cls_tokens, x), dim=1)  # CLS token as first token of every sequence
        # x = self.positional_encoding(x) # for positional encoding
        mask = torch.cat((torch.zeros(batch_size, 1, dtype=torch.bool, device=mask.device), mask), dim=1)  # Adjust mask for CLS token
        x = self.transformer(x, src_key_padding_mask=mask)
        output = self.regression_head(x[:, 0, :])
        return output.squeeze(-1)
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=20, verbose=True) # patience = #  of epochs before early stopping
        # scheduler picks diff. learning rates for different layers
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}

    def training_step(self, batch, batch_idx):
        self.train()
        inputs, targets, pad_mask = batch
        outputs = self(inputs, mask=pad_mask)
        loss = self.criterion(outputs, targets.squeeze(-1))
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        return {'loss': loss}
    
    def on_train_epoch_end(self):
        avg_loss = self.trainer.callback_metrics['train_loss']
        self.train_losses.append(avg_loss.item())
        self.log('avg_train_loss', avg_loss)

    def validation_step(self, batch, batch_idx):
        self.eval()
        inputs, targets, pad_mask = batch
        outputs = self(inputs, mask=pad_mask)
        with torch.no_grad():
            loss = self.criterion(outputs, targets.squeeze(-1))
        self.log('val_loss', loss, on_step=True, on_epoch=True)
        return {'loss': loss}
    
    def on_validation_epoch_end(self):
        avg_loss = self.trainer.callback_metrics['val_loss']
        self.val_losses.append(avg_loss.item())
        self.log('avg_val_loss', avg_loss)

    def plot_losses(self, fold, output_dir):
        print(f"Train Losses: {self.train_losses}")
        print(f"Val Losses: {self.val_losses}")
        plt.figure()
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.title(f'Fold {fold+1} Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(output_dir + f"Fold{fold+1}_loss_plot.png")
        plt.close()

    def reset_losses(self):
        self.train_losses = []
        self.val_losses = []

    def get_losses(self):
        return self.train_losses, self.val_losses



def get_predictions(model, dataloader):
    model.eval()
    all_actuals = []
    all_predictions = []
    with torch.no_grad():
        for batch in dataloader:
            process = psutil.Process(os.getpid())
            mem_before = process.memory_info().rss / (1024 ** 2)
            inputs, targets, pad_mask = batch
            outputs = model(inputs)
            all_actuals.extend(targets.cpu().numpy())
            all_predictions.extend(outputs.cpu().numpy())
            mem_after = process.memory_info().rss / (1024 ** 2)
            mem_used = mem_after - mem_before
            print(f"Memory used to get predictions: {mem_used:.2f} MB")
    return all_actuals, all_predictions

def compute_spearman_correlation(actuals, predictions):
    correlation, _ = spearmanr(actuals, predictions)
    return correlation

def train_and_evaluate(data, labels, groups, pad_mask, input_dim, seq_len, num_heads, hidden_dim, num_layers, batch_size, learning_rate, num_folds, max_epochs, early_stopping, output_dir):
    # kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    kf = GroupKFold(n_splits=num_folds)
    # logo = LeaveOneGroupOut()
    # logo.get_n_splits(data, labels, groups)
    # logo.get_n_splits(groups=groups) 
    fold_results = []
    fold_actuals = []
    fold_preds = []
    fold_train_actuals = []
    fold_train_preds = []
    fold_train_losses = []
    fold_val_losses = []

    train_spearmans = []
    val_spearmans = []
    model = TransformerRegressor(input_dim, seq_len, num_heads, hidden_dim, num_layers, lr=learning_rate)
    model.reset_losses()

    for fold, (train_idx, val_idx) in enumerate(kf.split(data, labels, groups)):
    # for fold, (train_idx, val_idx)  in enumerate(logo.split(data, labels, groups)):
        print(f'Fold {fold+1}/{num_folds}')
        train_data, val_data = data[train_idx], data[val_idx]
        train_labels, val_labels = labels[train_idx], labels[val_idx]
        train_pad_mask, val_pad_mask = pad_mask[train_idx], pad_mask[val_idx]

        train_dataloader = create_dataloader(train_data, train_labels, train_pad_mask, batch_size, is_train=1)
        val_dataloader = create_dataloader(val_data, val_labels, val_pad_mask, batch_size, is_train=0)

        if early_stopping: 
            trainer = Trainer(max_epochs=max_epochs, callbacks=[EarlyStopping(monitor="avg_val_loss", mode="min")], accelerator="auto", devices="auto", strategy="auto")
        else:
            trainer = Trainer(max_epochs=max_epochs, accelerator="auto", devices="auto", strategy="auto")
        trainer.fit(model, train_dataloader, val_dataloader) # try early stopping

        train_losses, val_losses = model.get_losses()
        fold_train_losses.append(train_losses)
        fold_val_losses.append(val_losses)

        val_actual, val_pred = get_predictions(model, val_dataloader)
        fold_actuals.append(val_actual)
        fold_preds.append(val_pred)

        train_actual, train_pred = get_predictions(model, train_dataloader)
        fold_train_actuals.append(train_actual)
        fold_train_preds.append(train_pred)

        train_spearmans.append(compute_spearman_correlation(train_actual, train_pred))
        val_spearmans.append(compute_spearman_correlation(val_actual, val_pred))

        fold_results.append(trainer.callback_metrics['val_loss'].item())

        model.plot_losses(fold, output_dir)

    print('Cross-Validation Results:')
    print(f'Mean Val Loss: {np.mean(fold_results)}, Std Dev Val Loss: {np.std(fold_results)}')

    best_fold = np.argmin(fold_results)
    print(f'Best Fold: {best_fold + 1}, Val Loss: {fold_results[best_fold]}')

    return model, fold_actuals, fold_preds, fold_train_actuals, fold_train_preds, fold_train_losses, fold_val_losses, fold_results, train_spearmans, val_spearmans


def plot_scatter(fold_actuals, fold_preds, fold_train_actuals, fold_train_preds, train_spearmans, val_spearmans, output_dir):
    num_folds = len(fold_actuals)
    for fold in range(num_folds):
        actuals_val = [float(x) for x in fold_actuals[fold]]
        preds_val = [float(x) for x in fold_preds[fold]]
        actuals_train = [float(x) for x in fold_train_actuals[fold]]
        preds_train = [float(x) for x in fold_train_preds[fold]]

        val_spearman = val_spearmans[fold]
        train_spearman = train_spearmans[fold]

        data_val = pd.DataFrame({'Actual': actuals_val, 'Predicted': preds_val, 'Type': 'Validation'})
        data_train = pd.DataFrame({'Actual': actuals_train, 'Predicted': preds_train, 'Type': 'Training'})
        
        data = pd.concat([data_val, data_train])
        
        data['Type'] = data['Type'].astype('category')
        
        # Determine dynamic limits
        min_val = min(data['Actual'].min(), data['Predicted'].min())
        max_val = max(data['Actual'].max(), data['Predicted'].max())
        
        g = sns.JointGrid(data=data, x='Actual', y='Predicted', height=10, xlim=(min_val, max_val), ylim=(min_val, max_val))
        
        sns.scatterplot(data=data[data['Type'] == 'Training'], x='Actual', y='Predicted', label='Training Data', ax=g.ax_joint, marker='o', s=10, alpha=0.6)
        sns.scatterplot(data=data[data['Type'] == 'Validation'], x='Actual', y='Predicted', label='Validation Data', ax=g.ax_joint, marker='o', s=10, alpha=0.6)

        g.ax_marg_x.hist(data_train['Actual'], bins=30, color='blue', alpha=0.6, density=True, label='Training')
        g.ax_marg_x.hist(data_val['Actual'], bins=30, color='orange', alpha=0.6, density=True, label='Validation')
        g.ax_marg_y.hist(data_train['Predicted'], bins=30, color='blue', alpha=0.6, density=True, orientation='horizontal')
        g.ax_marg_y.hist(data_val['Predicted'], bins=30, color='orange', alpha=0.6, density=True, orientation='horizontal')

        g.ax_joint.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        g.set_axis_labels('Observed Brain Size Residuals', 'Predicted Brain Size Residuals')
        
        # Add Spearman correlation annotations
        g.ax_joint.text(0.05, 0.95, f'Train Spearman: {train_spearman:.2f}', transform=g.ax_joint.transAxes, fontsize=12, verticalalignment='top', color='blue')
        g.ax_joint.text(0.05, 0.90, f'Val Spearman: {val_spearman:.2f}', transform=g.ax_joint.transAxes, fontsize=12, verticalalignment='top', color='orange')

        plt.suptitle(f' Encoder: Predicting Brainsize from ESM embeddings (Fold {fold+1})', y=1.02)
        g.ax_joint.legend(title='Data Type')
            
        
        plt.savefig(output_dir + f"Fold{fold+1}-plot.png")
        plt.close()

def eda(input_file, output_dir):
    with h5py.File(input_file, 'r') as f:
        x = torch.tensor(f['x'][:], dtype=torch.float32)
        y = torch.tensor(f['y'][:], dtype=torch.float32)
        mask = torch.tensor(f['mask'][:], dtype=torch.float32)
        species = [s.decode('utf-8') for s in f['species_labels'][:]]
        protein = [p.decode('utf-8') for p in f['protein_labels'][:]]
        order = [o.decode('utf-8') for o in f['order_labels'][:]]
    prot_df = pd.DataFrame(protein, columns=[0])
    prot_df["species_id"] = prot_df[0].apply(lambda x: x.split(":")[0])
    prot_df["uniprot_id"] = prot_df[0].apply(lambda x: x.split(":")[1])
    prot_df.drop(columns=[0], inplace=True)
    order_df = pd.DataFrame()
    order_df["order"]  = order
    order_df["brain_size_residuals"] = y
    order_dict = order_df.set_index("order")["brain_size_residuals"].to_dict()
    for sp in prot_df["species_id"].unique():
        print(sp, len(prot_df[prot_df["species_id"] == sp]))
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].hist(y.numpy(), bins=30, alpha=0.7, color='blue')
    axs[0].set_title("Brain Residuals")
    axs[0].set_xlabel("Brain Residual")
    axs[0].set_ylabel("Count")
    order_freq = Counter(order)
    bars = axs[1].bar(list(order_freq.keys()), list(order_freq.values()), alpha=0.7, color='red')
    axs[1].set_title("Order Frequency with Average Brain Size Residuals")
    axs[1].set_xlabel("Order")
    axs[1].set_ylabel("Count")
    axs[1].tick_params(axis='x', rotation=90)
    for bar, order_name in zip(bars, order_freq.keys()):
        height = bar.get_height()
        order_label = bar.get_x() + bar.get_width() / 2
        brain_residual = order_dict[order_name] 
        print(order_name)
        axs[1].annotate(f'{brain_residual:.2f}', xy=(order_label, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', rotation=45)
    plt.tight_layout()
    plt.show()
    plt.savefig(f"{output_dir}/{input_file.split('/')[-1].split('.')[0]}_freq.png")
    


if __name__ == "__main__":
    seed = 42 
    set_seed(seed) # set seed for reproducibility
    input_dim = 320 # ESM embedding shape
    num_heads = 8 # has to be a factor of input_dim
    hidden_dim = 128 # can increase for larger model
    num_layers = 2 # can increase for larger model
    batch_size = 10
    learning_rate = 1e-3
    num_folds = 10
    max_epochs = 100
    early_stopping = True

    # Change this based on if you're running on Farm etc. (you'll have to figure that out)
    device = torch.device('cuda')# torch.device('mps' if torch.backends.mps.is_available() else 'cpu') for MAC
    print(torch.cuda.is_available()) 
    # assert(False)
    # Initialize data
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024 ** 2)  # Convert to MB
    meta_df = pd.read_csv("/home/gluetown/brain/data/metadata.csv.gz", compression = "gzip")
    # input_file = "/home/gluetown/brain/outputs/encoder_cls_1/embeddings.h5"
    # output_dir = "/home/gluetown/brain/outputs/encoder_cls_1/graphs/"
    clusters = ["1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","25","26","27","28","29","30","31","32","33","34","35","36","37","38","39","40","41","42","43","44","45","46","47","48","49"]
    for cluster in clusters:
        input_file = f"/group/gquongrp/workspaces/claireh/brain/data/embeddings/test/go_terms/{cluster}.h5"
        output_dir = f"/home/gluetown/brain/data/embeddings/test/go_terms/graphs/cluster_{input_file.split('/')[-1].split('.')[0]}/"
        # input_file = "/home/gluetown/brain/data/embeddings/test/mmseqs/G1Q8L7.h5"
        # output_dir = f"/home/gluetown/brain/data/embeddings/test/mmseqs/graphs/{input_file.split('/')[-1].split('.')[0]}/run_hiddendim{hidden_dim}_num_layers{num_layers}_early_stopping_logo/"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        eda(input_file, output_dir)
        # x, y, mask = load_output(input_file)
        x, y, order, vocab, mask = load_output(input_file, order=True)
        mem_after = process.memory_info().rss / (1024 ** 2) 
        mem_used = mem_after - mem_before
        print(f"Memory used to load data: {mem_used:.2f} MB")

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # Convert to float32 and add cls token  
        x = x.to(torch.float32)
        y = y.to(torch.float32)

        # dataset parameters
        num_samples = x.shape[0]
        seq_len = x.shape[1]
        pad_mask = mask.to(torch.float32)
        cls_pad_column = torch.zeros(num_samples, 1).to(torch.float32)
        new_pad_mask = torch.cat((cls_pad_column, pad_mask), dim=1)
        data = x
        labels = y
        print(f"Data Shape: {data.shape}")
        print(f"Sequence Length: {seq_len}")       

        print(data.shape, labels.shape)

        model, fold_actuals, fold_preds, fold_train_actuals, fold_train_preds, fold_train_losses, fold_val_losses, fold_results, train_spearmans, val_spearmans = train_and_evaluate(
            data, labels, order, new_pad_mask, input_dim, seq_len, num_heads, hidden_dim, num_layers, batch_size, learning_rate, num_folds, max_epochs, early_stopping, output_dir
        )
        print(f"fold_actuals: {fold_actuals}")
        print(f"fold_preds: {fold_preds}")
        print(f"fold_train_actuals: {fold_train_actuals}")
        print(f"fold_train_preds: {fold_train_preds}")
        print(f"train_spearmans: {train_spearmans}")
        print(f"val_spearmans: {val_spearmans}")
        

        plot_scatter(fold_actuals, fold_preds, fold_train_actuals, fold_train_preds, train_spearmans, val_spearmans, output_dir)

        # Change directory etc. as you need to
        filename_train = output_dir + "model_train.txt"
        filename_val = output_dir + "model_val.txt"

        with open(filename_train, 'w') as filetr, open(filename_val, 'w') as filev:
            for i in range(len(train_spearmans)):
                filetr.write(f"{train_spearmans[i]}\n")
                filev.write(f"{val_spearmans[i]}\n")

        print((fold_actuals, fold_preds, fold_train_actuals, fold_train_preds, fold_train_losses, fold_val_losses))
        
