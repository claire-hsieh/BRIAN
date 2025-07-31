
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
import torch.optim as optim
import torch.nn.init as init
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
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

# Linear Attention
import sys
sys.path.append("/home/gluetown/brain/linear-attention-transformer/")
from local_attention import LocalAttention
from linformer import LinformerSelfAttention

from product_key_memory import PKM
from axial_positional_embedding import AxialPositionalEmbedding
from linear_attention_transformer.reversible import ReversibleSequence, SequentialSequence

from einops import rearrange, repeat
from linear_attention_transformer.linear_attention_transformer import *


def load_output(input_file, protein_labels = False):
    with h5py.File(input_file, 'r') as f:
        x = torch.tensor(f['x'][:], dtype=torch.float32)
        y = torch.tensor(f['y'][:], dtype=torch.float32)
        mask = torch.tensor(f['mask'][:], dtype=torch.float32)
        print(f"Loaded data from {input_file}")
        if protein_labels:
            # Load the nested list of protein labels
            group = f['protein_labels']
            protein_labels = [group[f'sublist_{i}'][:] for i in range(len(group))]
            return x, y, protein_labels, mask
        return x, y, mask

# DataLoader
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


class LinearAttentionTransformerLM(nn.Module):
    def __init__(
        self,
        num_tokens,
        dim,
        depth,
        max_seq_len,
        heads = 8,
        dim_head = 64,
        lr = 1e-3,
        causal = False,
        emb_dim = None,
        reversible = False,
        ff_chunks = 1,
        ff_glu = False,
        ff_dropout = 0.,
        attn_layer_dropout = 0.,
        attn_dropout = 0.,
        blindspot_size = 1,
        n_local_attn_heads = 0,
        local_attn_window_size = 128,
        return_embeddings = False,
        receives_context = False,
        pkm_layers = tuple(),
        pkm_num_keys = 128,
        attend_axially = False,
        linformer_settings = None,
        context_linformer_settings = None,
        use_axial_pos_emb = True,
        use_rotary_emb = False,
        shift_tokens = False
    ):
        assert n_local_attn_heads == 0 or (max_seq_len % local_attn_window_size) == 0, 'max sequence length must be divisible by the local attention window size'
        super().__init__()
        emb_dim = default(emb_dim, dim)
        self.max_seq_len = max_seq_len
        self.regression_head = nn.Linear(dim, 1)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim).to(torch.float32))
        self.transformer = LinearAttentionTransformer(dim, depth, max_seq_len, heads = heads, dim_head = dim_head, causal = causal, ff_chunks = ff_chunks, ff_glu = ff_glu, ff_dropout = ff_dropout, attn_layer_dropout = attn_layer_dropout, attn_dropout = attn_dropout, reversible = reversible, blindspot_size = blindspot_size, n_local_attn_heads = n_local_attn_heads, local_attn_window_size = local_attn_window_size, receives_context = receives_context, pkm_layers = pkm_layers, pkm_num_keys = pkm_num_keys, attend_axially = attend_axially, linformer_settings = linformer_settings, context_linformer_settings = context_linformer_settings, shift_tokens = shift_tokens)

        if emb_dim != dim:
            self.transformer = ProjectInOut(self.transformer, emb_dim, dim, project_out = not return_embeddings)

        self.norm = nn.LayerNorm(emb_dim)
        self.out = nn.Linear(emb_dim, num_tokens) if not return_embeddings else nn.Identity()
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

    def forward(self, x, mask, **kwargs):
        batch_size = x.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # CLS token expanded for every sequence in batch
        x = torch.cat((cls_tokens, x), dim=1)  # CLS token as first token of every sequence
        mask = torch.cat((torch.zeros(batch_size, 1, dtype=torch.bool, device=mask.device), mask), dim=1) 
        print(x.shape)
        x = self.transformer(x, **kwargs)
        x = self.norm(x)
        print(x.shape)
        output = self.regression_head(x[:, 0, :]) # pass CLS token into regression head
        print(output.shape)
        return output.squeeze(-1)
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=20, verbose=True)
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
            outputs = model(inputs, mask=pad_mask)
            all_actuals.extend(targets.cpu().numpy())
            all_predictions.extend(outputs.cpu().numpy())
            mem_after = process.memory_info().rss / (1024 ** 2)
            mem_used = mem_after - mem_before
            print(f"Memory used to get predictions: {mem_used:.2f} MB")
    return all_actuals, all_predictions

def compute_spearman_correlation(actuals, predictions):
    correlation, _ = spearmanr(actuals, predictions)
    return correlation

def train_and_evaluate(data, labels, pad_mask, input_dim, seq_len, num_heads, hidden_dim, num_layers, batch_size, learning_rate, num_folds, max_epochs, output_dir):
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    fold_results = []
    fold_actuals = []
    fold_preds = []
    fold_train_actuals = []
    fold_train_preds = []
    fold_train_losses = []
    fold_val_losses = []

    train_spearmans = []
    val_spearmans = []
    model = LinearAttentionTransformerLM(input_dim, seq_len, num_heads, hidden_dim, num_layers, lr=learning_rate)
    model.reset_losses()

    for fold, (train_idx, val_idx) in enumerate(kf.split(data)):
        print(f'Fold {fold+1}/{num_folds}')
        train_data, val_data = data[train_idx], data[val_idx]
        train_labels, val_labels = labels[train_idx], labels[val_idx]
        train_pad_mask, val_pad_mask = pad_mask[train_idx], pad_mask[val_idx]

        train_dataloader = create_dataloader(train_data, train_labels, train_pad_mask, batch_size, is_train=1)
        val_dataloader = create_dataloader(val_data, val_labels, val_pad_mask, batch_size, is_train=0)

        # Training Loop
        # trainer = pl.Trainer(max_epochs=max_epochs)
        # trainer.fit(model, train_dataloader, val_dataloader)

        # train_losses, val_losses = model.get_losses()
        # fold_train_losses.append(train_losses)
        # fold_val_losses.append(val_losses)

        # val_actual, val_pred = get_predictions(model, val_dataloader)
        # fold_actuals.append(val_actual)
        # fold_preds.append(val_pred)

        # train_actual, train_pred = get_predictions(model, train_dataloader)
        # fold_train_actuals.append(train_actual)
        # fold_train_preds.append(train_pred)
    
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


    


if __name__ == "__main__":
    seed = 42 
    set_seed(seed) # set seed for reproducibility
    input_dim = 320 # ESM embedding shape
    num_heads = 8 # has to be a factor of input_dim
    hidden_dim = 16 # can increase for larger model
    num_layers = 1 # can increase for larger model
    batch_size = 5
    learning_rate = 1e-3
    num_folds = 10
    max_epochs = 100

    # Change this based on if you're running on Farm etc. (you'll have to figure that out)
    # device = torch.device('cuda')
    torch.device('mps' if torch.backends.mps.is_available() else 'cpu') # for MAC

    # Initialize data
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024 ** 2)  # Convert to MB
    meta_df = pd.read_csv("/home/gluetown/brain/data/metadata.csv.gz", compression = "gzip")
    # input_file = "/home/gluetown/brain/outputs/encoder_cls_1/embeddings.h5"
    # output_dir = "/home/gluetown/brain/outputs/encoder_cls_1/graphs/"
    input_file = "/home/gluetown/brain/test_set/test/output/test_embeddings.h5" 
    output_dir = "/home/gluetown/brain/test_set/test/graphs/"
    x, y, mask = load_output(input_file)
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
        data, labels, new_pad_mask, input_dim, seq_len, num_heads, hidden_dim, num_layers, batch_size, learning_rate, num_folds, max_epochs, output_dir
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
