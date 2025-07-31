import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import AutoTokenizer, AutoModelForMaskedLM, EsmModel
from captum.attr import InputXGradient
import numpy as np
import random
import time
import pandas as pd
import h5py
from scipy.stats import spearmanr, pearsonr
from sklearn.model_selection import KFold, GroupKFold, LeaveOneGroupOut, train_test_split
from sklearn import metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, GroupKFold, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from pytorch_lightning import seed_everything
import psutil
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import statistics
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetTemperature, NVML_TEMPERATURE_GPU
import csv
import pickle as pkl
import json
sys.path.append("/home/gluetown/brain/scripts/scPointNet_dev/src/")
import scPointNet 
sys.path.append("/home/gluetown/brain/scripts/CLAM/models/")
from model_clam import *
from model_mil import *
sys.path.append("/group/gquongrp/workspaces/claireh/brain/scripts/mil_pytorch/src/")
import mil
import shutil
import requests
import io
import umap
from Bio import SeqIO
import h5py 

def log_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f"Memory Usage: {mem_info.rss / (1024 ** 2):.2f} MB")

def get_indices(element, lst):
    return [i for i in range(len(lst)) if lst[i] == element]

def flatten_list(lis):
    return [item for sublist in lis for item in sublist]

def add_lists(list1, list2):
    return [a + b for a, b in zip(list1, list2)]

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

def get_predictions(model, dataloader):
    model.eval()
    all_actuals = []
    all_predictions = []
    all_crit_idxs = []
    with torch.no_grad():
        for batch in dataloader:
            process = psutil.Process(os.getpid())
            mem_before = process.memory_info().rss / (1024 ** 2)
            inputs, targets, species = batch
            outputs, crit_idxs, A_feat, embedding = model(inputs)
            all_actuals.extend(targets.cpu().numpy())
            all_predictions.extend(outputs.cpu().numpy())
            all_crit_idxs.append(crit_idxs.cpu().numpy())
            mem_after = process.memory_info().rss / (1024 ** 2)
            mem_used = mem_after - mem_before
            print(f"Memory used to get predictions: {mem_used:.2f} MB")
        all_actuals = np.array(all_actuals)
        all_predictions = np.array(all_predictions)        
        return all_actuals, all_predictions, all_crit_idxs

def get_predictions_with_grad(model, dataloader, crit_idx = False):
    model.eval()
    all_actuals = []
    all_predictions = []
    all_crit_idxs = []
    all_gradients = []
    for batch in dataloader:
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / (1024 ** 2)
        inputs, targets, species = batch
        inputs.requires_grad = True
        if crit_idx: outputs, crit_idxs, A_feat, embedding = model(inputs)
        else: outputs  = model(inputs)
        outputs[0].backward(torch.ones_like(outputs[0]))  
        gradients = inputs.grad  
        input_x_grad = inputs * gradients
        all_actuals.extend(targets.cpu().detach().numpy())
        all_predictions.extend(outputs.cpu().detach().numpy())
        if crit_idx: all_crit_idxs.append(crit_idxs.cpu().detach().numpy())
        all_gradients.append(input_x_grad.cpu().detach().numpy())
        mem_after = process.memory_info().rss / (1024 ** 2)
        mem_used = mem_after - mem_before
        print(f"Memory used to get predictions: {mem_used:.2f} MB")
    all_actuals = np.array(all_actuals)
    all_predictions = np.array(all_predictions)
    if crit_idx: return all_actuals, all_predictions, crit_idxs
    else: return all_actuals, all_predictions 

def load_output(input_file, protein_labels = False, order = False, indiv_datasets = True):
    with h5py.File(input_file, 'r') as f:
        x = torch.tensor(f['x'][:], dtype=torch.float32)
        y = torch.tensor(f['y'][:], dtype=torch.float32)
        mask = torch.tensor(f['mask'][:], dtype=torch.float32)
        species_labels = [s.decode('utf-8') for s in f["species_labels"][:]]     
        # normalized_labels = torch.tensor(np.array([s.decode('utf-8') for s in f["normalized_labels"][:]], dtype=float))
        # standardized_labels = torch.tensor(np.array([s.decode('utf-8') for s in f["standardized_labels"][:]], dtype=float))
        print(f"Loaded data from {input_file}")
        if indiv_datasets:
            if protein_labels and order:
                with open(input_file.split(".h5")[0] + ".pkl", "rb") as fp:
                    protein_labels_dict = pkl.load(fp)
                order_labels_arr = f['order_labels'][:]
                if isinstance(order_labels_arr[0], bytes):
                    order_labels_arr = [label.decode('utf-8') for label in order_labels_arr]
                vocab = {label: i for i, label in enumerate(set(order_labels_arr))}
                order_labels = [vocab[label] for label in order_labels_arr]
                return x, y, order_labels_arr, vocab, mask, species_labels, protein_labels_dict
            if protein_labels:
                group = f['protein_labels']
                protein_labels = [group[f'sublist_{i}'][:] for i in range(len(group))]
                return x, y, protein_labels, mask
            if order:
                order_labels = f['order_labels'][:]
                if isinstance(order_labels[0], bytes):
                    order_labels = [label.decode('utf-8') for label in order_labels]
                vocab = {label: i for i, label in enumerate(set(order_labels))}
                order_labels = [vocab[label] for label in order_labels]
                return x, y, order_labels, vocab, mask, species_labels #, normalized_labels, standardized_labels
        else: 
            if protein_labels and order:
                protein_labels_arr = f['protein_labels'][:]
                if isinstance(protein_labels_arr[0], bytes):
                    protein_labels = [label.decode('utf-8') for label in protein_labels_arr]
                order_labels_arr = f['order_labels'][:]
                if isinstance(order_labels_arr[0], bytes):
                    order_labels_arr = [label.decode('utf-8') for label in order_labels_arr]
                vocab = {label: i for i, label in enumerate(set(order_labels_arr))}
                order_labels = [vocab[label] for label in order_labels_arr]
                return x, y, order_labels_arr, vocab, mask, species_labels, protein_labels
            if protein_labels:
                group = f['protein_labels']
                protein_labels = [group[f'sublist_{i}'][:] for i in range(len(group))]
                return x, y, protein_labels, mask
            if order:
                order_labels = f['order_labels'][:]
                if isinstance(order_labels[0], bytes):
                    order_labels = [label.decode('utf-8') for label in order_labels]
                vocab = {label: i for i, label in enumerate(set(order_labels))}
                order_labels = [vocab[label] for label in order_labels]
                return x, y, order_labels, vocab, mask, species_labels #, normalized_labels, standardized_labels
        return x, y, mask, species_labels #, normalized_labels, standardized_labels

# def load_output(input_file, protein_labels = False, order = False):
#     with h5py.File(input_file, 'r') as f:
#         x = torch.tensor(f['x'][:], dtype=torch.float32)
#         y = torch.tensor(f['y'][:], dtype=torch.float32)
#         mask = torch.tensor(f['mask'][:], dtype=torch.float32)
#         species_labels = [s.decode('utf-8') for s in f["species_labels"][:]]     
#         # normalized_labels = torch.tensor(np.array([s.decode('utf-8') for s in f["normalized_labels"][:]], dtype=float))
#         # standardized_labels = torch.tensor(np.array([s.decode('utf-8') for s in f["standardized_labels"][:]], dtype=float))
#         print(f"Loaded data from {input_file}")
#         if protein_labels:
#             group = f['protein_labels']
#             protein_labels = [group[f'sublist_{i}'][:] for i in range(len(group))]
#             return x, y, protein_labels, mask
#         if order:
#             order_labels = f['order_labels'][:]
#             if isinstance(order_labels[0], bytes):
#                 order_labels = [label.decode('utf-8') for label in order_labels]
#             vocab = {label: i for i, label in enumerate(set(order_labels))}
#             order_labels = [vocab[label] for label in order_labels]
#             return x, y, order_labels, vocab, mask, species_labels #, normalized_labels, standardized_labels
#         return x, y, mask, species_labels #, normalized_labels, standardized_labels


def initialize_dataset(pad_x, y, order_labels, vocab, mask, species_labels):
    x_list = []
    for x in remove_mask(pad_x, mask):
        x_list.append(x.permute(1,0))
    trouble_makers = ["UP000189704_1868482.fasta", "UP000009136_9913.fasta", "UP000694520_30521.fasta"]
    troublemaker_indices = list(np.array([get_indices(i, species_labels) for i in trouble_makers]).flatten())
    y_list = list(y)    
    # swapped = {v:k for k, v in vocab.items()}
    all_orders = [vocab[k] for k in order_labels]
    order_list = [i if Counter(all_orders)[i] > 5 else "Other" for i in all_orders]    
    # order_list = [order_list[i] for i in range(len(order_list)) if i not in troublemaker_indices]
    # species_labels = [species_labels[i] for i in range(len(species_labels)) if i not in troublemaker_indices]
    # temporary solution to species - embeddings mismatch: 
    x_list = [x_list[i] for i in range(len(x_list)) if i not in troublemaker_indices]
    y_list_normalized = min_max_normalize(y_list)
    y_list_standardized = z_score_normalize(y_list)
    return x_list, y_list, y_list_normalized, y_list_standardized, order_list, mask, species_labels


def remove_mask(x, mask):
    # returns iterable of x (ragged)
    for species in range(mask.shape[0]): 
        bool_array = [True if i == 1 else False for i in mask[species]]
        yield x[species][bool_array]

class ESM2_Dataset(Dataset):
    def __init__(self, x, y, mask):# , species_id):
        self.x = x 
        self.y = y 
        self.mask = mask
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], mask[idx]
   
# def create_dataloader(data, labels, pad_mask, batch_size, is_train):
#     dataset = ESM2_Dataset(data, labels, pad_mask)
#     if is_train:
#         return DataLoader(dataset, batch_size=batch_size, shuffle=True)
#     else:
#         return DataLoader(dataset, batch_size=batch_size, shuffle=False)

class ESMPointCloudDataset(Dataset):
    def __init__(self, point_cloud_dict, labels_dict):
        self.point_cloud_dict = point_cloud_dict
        self.labels_dict = labels_dict
        self.keys = list(point_cloud_dict.keys())
    def __len__(self):
        return len(self.keys)    
    def __getitem__(self, idx):
        key = self.keys[idx]
        point_cloud = self.point_cloud_dict[key]
        label = self.labels_dict[key]
        return point_cloud, label

class ListDataset(Dataset):
    def __init__(self, x, y, species):
        self.x = x
        self.y = y
        self.species = species
    def __len__(self):
        return(len(self.x))
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.species[idx]
        
# def collate_mil(batch):
#     # Collate function to handle variable-sized point clouds
#     batch_data = []
#     batch_labels = []
#     for point_cloud, label in batch:
#         batch_data.append(point_cloud)
#         batch_labels.append(label)    
#     # Pad point clouds to the same length
#     max_len = max([pc.shape[0] for pc in batch_data])
#     padded_data = torch.zeros((len(batch_data), max_len, 320))
#     for i, pc in enumerate(batch_data):
#         padded_data[i, :pc.shape[0], :] = pc    
#     # Stack the padded point clouds into a 3D tensor
#     stacked_data = torch.stack([padded_data[i] for i in range(len(batch_data))])
#     stacked_data = stacked_data.permute(0,2,1) # to match point net input (species, 320, genes)
#     batch_labels = torch.tensor(batch_labels)
#     return stacked_data, batch_labels

# def create_dataloader(data, labels, batch_size, is_train):
#     dataset = ESMPointCloudDataset(data, labels)
#     if is_train: 
#         return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_mil)
#     else:
#         return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_mil)

def create_dataloader(data, labels, species, batch_size, is_train):
    dataset = ListDataset(data, labels, species)
    if is_train: 
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    else:
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)


class PointNetRegHead(nn.Module):
    def __init__(self, first_dim=40, second_dim = 64, conv1d_dims=[64], fc_blocks=[256], global_features=256, k=1, num_layers = 1, use_dropout=True, use_layer_norm=False):
        super(PointNetRegHead, self).__init__()
        self.backbone = scPointNet.PointNetBackbone(first_dim, second_dim, conv1d_dims=conv1d_dims, fc_blocks=fc_blocks, global_features = global_features)
        self.linear = nn.Linear(global_features, 256)
        # layers = []
        # output_dim = global_features
        # for i in range(num_layers):
        #     input_dim = output_dim
        #     output_dim = global_features // (i + 1)
        #     layers.append(nn.Linear(k, input_dim))
        #     layers.append(nn.ReLU())
        #     print(input_dim)
        # self.hidden_layers = nn.Sequential(*layers)
        self.out = nn.Linear(256, k)
        # batchnorm for the first linear layers
        self.bn = nn.BatchNorm1d(global_features)
        self.dropout = nn.Dropout(p=0.2)
        self.layer_norm = nn.LayerNorm(global_features)
        self.use_dropout = use_dropout
        self.use_layer_norm = use_layer_norm
        # The paper states that batch norm was only added to the layer 
        # before the classification layer, but another version adds dropout  
        # self.dropout = nn.Dropout(p=0.2)
    def forward(self, x, mask=None):
        print(f"x: {x.shape}")
        x, crit_idxs, A_feat = self.backbone(x, mask) 
        # x = self.hidden_layers(x)
        # print(f"x: {x.shape} after backbone")
        x = F.relu(self.linear(x))
        # print(f"x: {x.shape} after relu")
        if self.use_layer_norm: x = self.layer_norm(x)
        # print(f"x: {x.shape} after layer norm")
        if self.use_dropout: x = self.dropout(x)
        # print(f"x: {x.shape} after dropout")
        embedding = x.clone()
        x = self.out(x)        
        # print(f"x: {x.shape} after last linear layer")
        return x, crit_idxs, A_feat, embedding

# x: torch.Size([1, 320, 13441])
# x: torch.Size([1, 256]) after backbone
# x: torch.Size([1, 256]) after relu
# x: torch.Size([1, 256]) after dropout
# x: torch.Size([1, 1]) after last linear layer

class PointNetRegHead2(nn.Module):
    def __init__(self, first_dim=40, second_dim = 64, conv1d_dims=[64], fc_blocks=[256], global_features=256, k=1, num_layers = 1, use_dropout=True, use_layer_norm=False):
        super(PointNetRegHead2, self).__init__()
        self.backbone = scPointNet.PointNetBackbone(first_dim, second_dim, conv1d_dims=conv1d_dims, fc_blocks=fc_blocks, global_features = global_features)
        self.linear = nn.Linear(global_features, 256)
        self.out = nn.Linear(256, k)
        self.bn = nn.BatchNorm1d(global_features)
        self.dropout = nn.Dropout(p=0.2)
        self.layer_norm = nn.LayerNorm(global_features)
        self.use_dropout = use_dropout
        self.use_layer_norm = use_layer_norm
    def forward(self, x, mask=None):
        print(f"x: {x.shape}")
        x, crit_idxs, A_feat = self.backbone(x, mask) 
        x = F.relu(self.linear(x))
        if self.use_layer_norm: x = self.layer_norm(x)
        if self.use_dropout: x = self.dropout(x)
        embedding = x.clone()
        x = self.out(x)
        return x



class Encoder_PointNet(nn.Module):
    def __init__(self, input_dim, input_batch_num, hidden_layer=[900, 40], layernorm=True, activation=nn.ReLU(), batchnorm=False, dropout_rate=0, add_linear_layer=False, clip_threshold=None,
                 output_dim=1, global_features=256):
        super(Encoder_PointNet, self).__init__()
        self.encoder = scPointNet.Encoder_SC(input_dim, input_batch_num, hidden_layer, layernorm, activation, batchnorm, dropout_rate, add_linear_layer, clip_threshold)
        self.pointnet = PointNetRegHead(hidden_layer[1],global_features=global_features, k=output_dim)
    def forward(self, x):
        x = self.encoder(x, None).permute(0, 2, 1)
        return self.pointnet(x)

# model = Encoder_PointNet(input_dim, input_batch_num = 0, global_features = hidden_dim, hidden_layer = [320, 40])

class StackedEsmPointnet(nn.Module):
    def __init__(self, first_dim=320, second_dim = 64, conv1d_dims=[64], fc_blocks=[256], global_features=256, k=1, use_dropout=True, use_layer_norm=False):
        super(StackedEsmPointnet, self).__init__()
        self.pointNet = PointNetRegHead2(first_dim, second_dim, conv1d_dims, fc_blocks, global_features, k, use_dropout, use_layer_norm)
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
        # self.esm_model = AutoModelForMaskedLM.from_pretrained("facebook/esm2_t6_8M_UR50D")
        self.esm_model = EsmModel.from_pretrained("facebook/esm2_t6_8M_UR50D")
    def forward(self, x):
        # tokenized = self.tokenizer(x,return_tensors="pt")
        # better to tokenize before passing in, b/c captum only takes in tensors
        x = x.long()
        embedding = self.esm_model(x) # ['mean_representations'][6] # not sure about this step
        embedding = embedding.last_hidden_state.permute(0,2,1)
        out = self.pointNet(embedding)
        return out



# def plot_losses(fold_train_losses, fold_val_losses, species_list, output_dir):
#     for fold in range(len(fold_train_losses)):
#         print(f"Train Losses: {fold_train_losses[fold]}")
#         print(f"Val Losses: {fold_val_losses[fold]}")
#         plt.figure()
#         plt.plot(fold_train_losses[fold], label='Training Loss')
#         plt.plot(fold_val_losses[fold], label='Validation Loss')
#         plt.title(f'Fold {fold+1} Training and Validation Loss | Leave Out {species_list[fold]}')
#         plt.xlabel('Epoch')
#         plt.ylabel('Loss')
#         plt.legend()
#         if not os.path.exists(output_dir):
#             os.makedirs(output_dir)
#         plt.savefig(output_dir + f"Fold{fold+1}_loss_plot.png")
#         plt.close()



def plot_epoch_losses(epoch_train_losses, epoch_val_losses, species, output_dir):
    print(f"Train Losses: {epoch_train_losses}")
    print(f"Val Losses: {epoch_val_losses}")
    plt.figure()
    plt.plot(epoch_train_losses, label='Training Loss')
    plt.plot(epoch_val_losses, label='Validation Loss')
    plt.title(f'Training and Validation Loss | Leave Out {species}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(output_dir + f"Fold{species}_loss_plot.png")
    plt.close()

def compute_spearman_correlation(actuals, predictions):
    correlation, _ = spearmanr(actuals, predictions)
    return correlation

def compute_pearson_correlation(actuals, predictions):
    if isinstance(actuals, torch.Tensor):
        actuals = actuals.cpu().numpy()
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()    
    correlation, _ = pearsonr(actuals, predictions)
    return correlation

def mean(ls):
    return sum(ls) / len(ls)

class EarlyStopping:
    def __init__(self, patience=15, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_model_state = None
    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.best_model_state = model.state_dict()
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_state = model.state_dict()
            self.counter = 0
    def load_best_model(self, model):
        model.load_state_dict(self.best_model_state)
    
def train_and_evaluate_kfold(data, labels, group, species_list, input_dim, output_dim, hidden_dim, batch_size, learning_rate, num_folds, max_epochs, use_early_stopping, use_dropout,num_layers, output_dir):
    model_output_dir = output_dir + "model/"
    if not os.path.exists(model_output_dir):
        os.mkdir(model_output_dir)
    # Check gpu usage
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"GPU Utilization: {torch.cuda.utilization(device)}%")
        print(f"Memory Allocated: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")
        print(f"Memory Cached: {torch.cuda.memory_reserved(device) / 1024**2:.2f} MB")
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(0)
        temperature = nvmlDeviceGetTemperature(handle, NVML_TEMPERATURE_GPU)
        print(f"GPU Temperature: {temperature} C")
    else:
        device = torch.device("cpu")
        print("CUDA is not available.")
    # Move data to gpu (if available)
    data = [torch.tensor(d, dtype=torch.float32).to(device) for d in data]
    labels = [torch.tensor(l, dtype=torch.float32).to(device) for l in labels]
    loss_fn = nn.MSELoss()  
    # training loop
    fold_actuals = []
    fold_preds = []
    fold_train_actuals = []
    fold_train_preds = []
    fold_train_losses = []
    fold_val_losses = []    
    train_spearmans = []
    val_spearmans = []
    train_pearsons = []
    val_pearsons = []
    train_num_genes = []
    all_train_labels = []
    val_num_genes = []
    all_val_labels = []
    crit_indices = {"train": {}, "val": {}}
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    # kf = GroupKFold(n_splits=num_folds)
    # for fold, (train_idx, val_idx) in enumerate(kf.split(data, labels, group)):
    # for fold, species in enumerate(species_list):
    for fold, (train_idx, val_idx) in enumerate(kf.split(data)):
        filename = fold # change to species if using leave one out
        model = PointNetRegHead(first_dim=input_dim, global_features = hidden_dim, k = output_dim, num_layers = num_layers, use_dropout=use_dropout).to(device)
        # model = Encoder_PointNet(input_dim, input_batch_num = 0, global_features = hidden_dim, hidden_layer = [input_dim, 40]).to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        if use_early_stopping:
            early_stopping = EarlyStopping(patience=30, delta=0.01)
        print(f'Fold {fold+1}/{len(list(set(group)))}')
        log_memory_usage()  # Log memory usage at the start of each fold
        # Group K Fold 
        train_data, val_data = [data[i] for i in train_idx], [data[i] for i in val_idx]
        train_labels, val_labels = [labels[i] for i in train_idx], [labels[i] for i in val_idx]
        train_group, val_group = [group[i] for i in train_idx], [group[i] for i in val_idx]
        train_species, val_species = [species_list[i] for i in train_idx], [species_list[i] for i in val_idx]
        # LOGO 
        # train_idx = get_indices(species, group)
        # train_data, val_data = [data[i] for i in range(len(data)) if i not in train_idx], [data[i] for i in range(len(data)) if i in train_idx] 
        # train_labels, val_labels = [labels[i] for i in range(len(data)) if i not in train_idx], [labels[i] for i in range(len(labels)) if i in train_idx] 
        # train_species, val_species = [group[i] for i in range(len(data)) if i not in train_idx], [group[i] for i in range(len(data)) if i in train_idx]
        train_dataloader = create_dataloader(train_data, train_labels, train_species, 1, is_train=1)
        val_dataloader = create_dataloader(val_data, val_labels, val_species, 1, is_train=0)
        train_actual = []
        train_pred = []
        val_actual = []
        val_pred = []
        epoch_train_losses = []
        epoch_val_losses = []
        for epoch in tqdm(range(max_epochs)): 
            model.train()                       
            train_losses = []
            val_losses = []         
            train_crit_idxs = {}    
            val_crit_idxs = {}
            for batch_x, batch_y, batch_species in train_dataloader: # just loading things in 1 by 1 (not actually batching)
                optimizer.zero_grad()
                # pred_y_list = [model(x.unsqueeze(0))[0] for x in batch_x]
                # pred_y = torch.cat(pred_y_list, dim=0)
                pred_y, crit_idx, A_feat, embedding = model(batch_x)
                train_crit_idxs[batch_species[0]] = crit_idx
                loss = loss_fn(pred_y.squeeze(1), batch_y)
                train_losses.append(loss.item())                
                loss.backward() 
                optimizer.step()
            model.eval()
            with torch.no_grad():
                for batch_x, batch_y, batch_species in val_dataloader:
                    # pred_y, crit_idxs, A_feat, embedding = model(batch_x)
                    # pred_y_list = [model(x.unsqueeze(0))[0] for x in batch_x]
                    # pred_y = torch.cat(pred_y_list, dim=0)
                    pred_y, crit_idx, A_feat, embedding = model(batch_x)
                    val_crit_idxs[batch_species[0]] = crit_idx
                    loss = loss_fn(pred_y.squeeze(1), batch_y)
                    val_losses.append(loss.item())
                epoch_train_losses.append(mean(train_losses))
                epoch_val_losses.append(mean(val_losses))
                if use_early_stopping:
                    early_stopping(mean(val_losses), model)
                    if early_stopping.early_stop:
                        print(f"Early stopping at epoch {epoch}")
                        early_stopping.load_best_model(model)
                        break
        crit_indices["train"][fold] = {k: v.cpu().numpy().tolist() for k, v in train_crit_idxs.items()}
        crit_indices["val"][fold] = {k: v.cpu().numpy().tolist() for k, v in val_crit_idxs.items()}
        # if device == "cuda":
        #     train_number_of_genes = [i.cpu().shape[1] for train_number_of_genes in train_data]
        #     val_number_of_genes = [i.cpu().shape[1] for i in val_data]
        # else:
        #     train_number_of_genes = [i.shape[1] for train_number_of_genes in train_data]
        #     val_number_of_genes = [i.shape[1] for i in val_data]
        # train_corr, _ = pearsonr(train_number_of_genes, train_labels.cpu())
        # val_corr, _ = pearsonr(val_number_of_genes, val_labels.cpu())
        train_num_genes.append([i.shape[1] for i in train_data])
        all_train_labels.append(train_labels)
        val_num_genes.append([i.shape[1] for i in val_data])
        all_val_labels.append(val_labels)
        fold_train_losses.append(epoch_train_losses)
        fold_val_losses.append(epoch_val_losses)
        val_actual, val_pred, val_crit_idxs = get_predictions(model, val_dataloader)
        fold_actuals.append(val_actual)
        fold_preds.append(val_pred.flatten())
        train_actual, train_pred, train_crit_idxs = get_predictions(model, train_dataloader)
        fold_train_actuals.append(train_actual)
        fold_train_preds.append(train_pred.flatten())            
        train_spearmans.append(compute_spearman_correlation(train_actual, train_pred))
        val_spearmans.append(compute_spearman_correlation(val_actual, val_pred))
        train_pearsons.append(compute_pearson_correlation(train_actual, train_pred.flatten())) 
        val_pearsons.append(compute_pearson_correlation(val_actual, val_pred.flatten()))
        correlations = [compute_spearman_correlation(train_actual, train_pred), compute_spearman_correlation(val_actual, val_pred), compute_pearson_correlation(train_actual, train_pred.flatten()), compute_pearson_correlation(val_actual, val_pred.flatten())]
        plot_epoch_losses(epoch_train_losses, epoch_val_losses, filename, output_dir)
        plot_fold_scatter(val_actual, val_pred.flatten(), train_actual, train_pred.flatten(), correlations, train_group, val_group, filename, output_dir)
        # can't use sometimes w/ logo b/c some orders only have 1 member and pearson doesn't allow calc. on single values
        torch.save(model.state_dict(), f"{model_output_dir}/{filename}_{fold}_model.pth")
        log_memory_usage()  # Log memory usage at the end of each fold  
    # plot_scatter(fold_actuals, fold_preds, fold_train_actuals, fold_train_preds, train_spearmans, val_spearmans, output_dir+species+"/")
    # plot_losses(fold_train_losses, fold_val_losses, species_list, output_dir+species+"/")
    # save model
    return model, fold_actuals, fold_preds, fold_train_actuals, fold_train_preds, fold_train_losses, fold_val_losses, train_spearmans, val_spearmans,  train_num_genes, all_train_labels, val_num_genes, all_val_labels, crit_indices


def train_and_evaluate_logo(data, labels, group, species_list, input_dim, output_dim, hidden_dim, batch_size, learning_rate, num_folds, max_epochs, use_early_stopping, use_dropout, output_dir):
    model_output_dir = output_dir + "model/"
    if not os.path.exists(model_output_dir):
        os.mkdir(model_output_dir)
    # Check gpu usage
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"GPU Utilization: {torch.cuda.utilization(device)}%")
        print(f"Memory Allocated: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")
        print(f"Memory Cached: {torch.cuda.memory_reserved(device) / 1024**2:.2f} MB")
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(0)
        temperature = nvmlDeviceGetTemperature(handle, NVML_TEMPERATURE_GPU)
        print(f"GPU Temperature: {temperature} C")
    else:
        device = torch.device("cpu")
        print("CUDA is not available.")
    # Move data to gpu (if available)
    data = [torch.tensor(d, dtype=torch.float32).to(device) for d in data]
    labels = [torch.tensor(l, dtype=torch.float32).to(device) for l in labels]
    loss_fn = nn.MSELoss()  
    # training loop
    fold_actuals = []
    fold_preds = []
    fold_train_actuals = []
    fold_train_preds = []
    fold_train_losses = []
    fold_val_losses = []    
    train_spearmans = []
    val_spearmans = []
    train_pearsons = []
    val_pearsons = []
    crit_indices = {"train": {}, "val": {}}
    for fold, species in enumerate(species_list):
        # train_corr = compute_pearson_correlation([i.shape[1] for i in train_data], train_labels)
        # val_corr = compute_pearson_correlation([i.shape[1] for i in val_data], val_labels)
        filename = species 
        model = PointNetRegHead(first_dim=input_dim, global_features = hidden_dim, k = output_dim, use_dropout=use_dropout).to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        if use_early_stopping:
            early_stopping = EarlyStopping(patience=30, delta=0.01)
        print(f'Fold {fold+1}/{len(species_list)}')
        log_memory_usage()  # Log memory usage at the start of each fold
        # LOGO 
        train_idx = get_indices(species, group)
        train_data, val_data = [data[i] for i in range(len(data)) if i not in train_idx], [data[i] for i in range(len(data)) if i in train_idx] 
        train_labels, val_labels = [labels[i] for i in range(len(data)) if i not in train_idx], [labels[i] for i in range(len(labels)) if i in train_idx] 
        train_species, val_species = [group[i] for i in range(len(data)) if i not in train_idx], [group[i] for i in range(len(data)) if i in train_idx]
        train_dataloader = create_dataloader(train_data, train_labels, 1, is_train=1)
        val_dataloader = create_dataloader(val_data, val_labels, 1, is_train=0)
        train_actual = []
        train_pred = []
        val_actual = []
        val_pred = []
        epoch_train_losses = []
        epoch_val_losses = []
        for epoch in tqdm(range(max_epochs)): 
            model.train()                       
            train_losses = []
            val_losses = []             
            for batch_x, batch_y in train_dataloader: # just loading things in 1 by 1 (not actually batching)
                optimizer.zero_grad()
                # pred_y_list = [model(x.unsqueeze(0))[0] for x in batch_x]
                # pred_y = torch.cat(pred_y_list, dim=0)
                pred_y, train_crit_idxs, A_feat, embedding = model(batch_x)
                loss = loss_fn(pred_y.squeeze(1), batch_y)
                train_losses.append(loss.item())                
                loss.backward() 
                optimizer.step()
            model.eval()
            with torch.no_grad():
                for batch_x, batch_y in val_dataloader:
                    # pred_y, crit_idxs, A_feat, embedding = model(batch_x)
                    # pred_y_list = [model(x.unsqueeze(0))[0] for x in batch_x]
                    # pred_y = torch.cat(pred_y_list, dim=0)
                    pred_y, pred_crit_idxs, A_feat, embedding = model(batch_x)
                    loss = loss_fn(pred_y.squeeze(1), batch_y)
                    val_losses.append(loss.item())
                epoch_train_losses.append(mean(train_losses))
                epoch_val_losses.append(mean(val_losses))
                if use_early_stopping:
                    early_stopping(mean(val_losses), model)
                    if early_stopping.early_stop:
                        print(f"Early stopping at epoch {epoch}")
                        early_stopping.load_best_model(model)
                        break
        fold_train_losses.append(epoch_train_losses)
        fold_val_losses.append(epoch_val_losses)
        val_actual, val_pred = get_predictions(model, val_dataloader)
        fold_actuals.append(val_actual)
        fold_preds.append(val_pred.flatten())
        train_actual, train_pred = get_predictions(model, train_dataloader)
        fold_train_actuals.append(train_actual)
        fold_train_preds.append(train_pred.flatten())            
        train_spearmans.append(compute_spearman_correlation(train_actual, train_pred))
        val_spearmans.append(compute_spearman_correlation(val_actual, val_pred))
        # train_pearsons.append(compute_pearson_correlation(train_actual, train_pred.flatten())) 
        # val_pearsons.append(compute_pearson_correlation(val_actual, val_pred.flatten()))
        correlations = [compute_spearman_correlation(train_actual, train_pred), compute_spearman_correlation(val_actual, val_pred), compute_pearson_correlation(train_actual, train_pred.flatten()), compute_pearson_correlation(val_actual, val_pred.flatten())]
        # correlations = [compute_spearman_correlation(train_actual, train_pred), compute_spearman_correlation(val_actual, val_pred)]
        plot_epoch_losses(epoch_train_losses, epoch_val_losses, filename, output_dir)
        plot_fold_scatter(val_actual, val_pred.flatten(), train_actual, train_pred.flatten(), correlations, train_species, val_species, filename, output_dir)
        # can't use sometimes w/ logo b/c some orders only have 1 member and pearson doesn't allow calc. on single values
        torch.save(model.state_dict(), f"{model_output_dir}/{filename}_{fold}_model.pth")
        log_memory_usage()  # Log memory usage at the end of each fold  
    # plot_scatter(fold_actuals, fold_preds, fold_train_actuals, fold_train_preds, train_spearmans, val_spearmans, output_dir+species+"/")
    # plot_losses(fold_train_losses, fold_val_losses, species_list, output_dir+species+"/")
    # save model
    return model, fold_actuals, fold_preds, fold_train_actuals, fold_train_preds, fold_train_losses, fold_val_losses, train_spearmans, val_spearmans, 


# def plot_scatter(fold_actuals, fold_preds, fold_train_actuals, fold_train_preds, train_spearmans, val_spearmans, output_dir):
#     num_folds = len(fold_actuals)
#     for fold in range(num_folds):
#         actuals_val = [float(x) for x in fold_actuals[fold]]
#         preds_val = [float(x) for x in fold_preds[fold]]
#         actuals_train = [float(x) for x in fold_train_actuals[fold]]
#         preds_train = [float(x) for x in fold_train_preds[fold]]
#         train_spearman = train_spearmans[fold]
#         # train_pearson = train_pearsons[fold]
#         # val_pearson = val_pearsons[fold]
#         val_spearman = val_spearmans[fold]
#         data_val = pd.DataFrame({'Actual': actuals_val, 'Predicted': preds_val, 'Type': 'Validation'})
#         data_train = pd.DataFrame({'Actual': actuals_train, 'Predicted': preds_train, 'Type': 'Training'})
#         data = pd.concat([data_val, data_train])        
#         data['Type'] = data['Type'].astype('category')        
#         # Determine dynamic limits
#         min_val = min(data['Actual'].min(), data['Predicted'].min())
#         max_val = max(data['Actual'].max(), data['Predicted'].max())        
#         g = sns.JointGrid(data=data, x='Actual', y='Predicted', height=10, xlim=(min_val, max_val), ylim=(min_val, max_val))
#         sns.scatterplot(data=data[data['Type'] == 'Training'], x='Actual', y='Predicted', label='Training Data', ax=g.ax_joint, marker='o', s=10, alpha=0.6)
#         sns.scatterplot(data=data[data['Type'] == 'Validation'], x='Actual', y='Predicted', label='Validation Data', ax=g.ax_joint, marker='o', s=10, alpha=0.6)
#         g.ax_marg_x.hist(data_train['Actual'], bins=30, color='blue', alpha=0.6, density=True, label='Training')
#         g.ax_marg_x.hist(data_val['Actual'], bins=30, color='orange', alpha=0.6, density=True, label='Validation')
#         g.ax_marg_y.hist(data_train['Predicted'], bins=30, color='blue', alpha=0.6, density=True, orientation='horizontal')
#         g.ax_marg_y.hist(data_val['Predicted'], bins=30, color='orange', alpha=0.6, density=True, orientation='horizontal')
#         g.ax_joint.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
#         g.set_axis_labels('Observed Brain Size Residuals', 'Predicted Brain Size Residuals')
#         # Add Spearman correlation annotations
#         g.ax_joint.text(0.05, 0.95, f'Train Spearman: {train_spearman:.2f}', transform=g.ax_joint.transAxes, fontsize=12, verticalalignment='top', color='blue')
#         # g.ax_joint.text(0.05, 0.90, f'Train Pearson: {train_pearson:.2f}', transform=g.ax_joint.transAxes, fontsize=12, verticalalignment='top', color='blue')
#         # g.ax_joint.text(0.05, 0.90, f'Train mean: {statistics.mean(data[data["Type"] == "Training"]["Predicted"]):.2f}', transform=g.ax_joint.transAxes, fontsize=12, verticalalignment='top', color='blue')
#         # g.ax_joint.text(0.05, 0.85, f'Train Std Dev: {statistics.stdev(data[data["Type"] == "Training"]["Predicted"]):.2f}', transform=g.ax_joint.transAxes, fontsize=12, verticalalignment='top', color='blue')
#         g.ax_joint.text(0.05, 0.80, f'Val Spearman: {val_spearman:.2f}', transform=g.ax_joint.transAxes, fontsize=12, verticalalignment='top', color='orange')
#         # g.ax_joint.text(0.05, 0.75, f'Val Pearson: {val_pearson:.2f}', transform=g.ax_joint.transAxes, fontsize=12, verticalalignment='top', color='orange')
#         # g.ax_joint.text(0.05, 0.75, f'Val mean: {statistics.mean(data[data["Type"] == "Validation"]["Predicted"]):.2f}', transform=g.ax_joint.transAxes, fontsize=12, verticalalignment='top', color='orange')
#         # g.ax_joint.text(0.05, 0.70, f'Val Std Dev: {statistics.stdev(data[data["Type"] == "Validation"]["Predicted"]):.2f}', transform=g.ax_joint.transAxes, fontsize=12, verticalalignment='top', color='orange')
#         plt.suptitle(f' Encoder: Predicting Brainsize from ESM embeddings (Fold {fold+1})', y=1.02)
#         g.ax_joint.legend(title='Data Type')    
#         if not os.path.exists(output_dir):
#             os.makedirs(output_dir)     
#         plt.savefig(output_dir + f"Fold{fold+1}-plot.png")
#         plt.close()


def plot_fold_scatter(actuals_val, preds_val, actuals_train, preds_train, correlations, train_species, val_species, species , output_dir, detailed=False,train_corr=0, val_corr=0):
    data_val = pd.DataFrame({'Actual': actuals_val, 'Predicted': preds_val, 'Type': 'Validation', 'SpeciesOrder': val_species})
    data_train = pd.DataFrame({'Actual': actuals_train, 'Predicted': preds_train, 'Type': 'Training', 'SpeciesOrder': train_species})
    data = pd.concat([data_val, data_train])
    data['Type'] = data['Type'].astype('category')
    # Determine dynamic limits
    min_val = min(data['Actual'].min(), data['Predicted'].min())
    max_val = max(data['Actual'].max(), data['Predicted'].max())     
    g = sns.JointGrid(data=data, x='Actual', y='Predicted', height=10, xlim=(min_val, max_val), ylim=(min_val, max_val))
    sns.scatterplot(data=data[data['Type'] == 'Training'], x='Actual', y='Predicted', label='Training Data', ax=g.ax_joint, marker='o', s=10, alpha=0.6)
    sns.scatterplot(data=data[data['Type'] == 'Validation'], x='Actual', y='Predicted', hue='SpeciesOrder', style='SpeciesOrder', palette='Oranges', ax=g.ax_joint, s=20, alpha=0.6)
    g.ax_marg_x.hist(data_train['Actual'], bins=30, color='blue', alpha=0.6, density=True, label='Training')
    g.ax_marg_x.hist(data_val['Actual'], bins=30, color='orange', alpha=0.6, density=True, label='Validation')
    g.ax_marg_y.hist(data_train['Predicted'], bins=30, color='blue', alpha=0.6, density=True, orientation='horizontal')
    g.ax_marg_y.hist(data_val['Predicted'], bins=30, color='orange', alpha=0.6, density=True, orientation='horizontal')
    # Reference line
    g.ax_joint.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    g.set_axis_labels('Observed Brain Size Residuals', 'Predicted Brain Size Residuals')
    # Add Spearman correlation annotations
    train_spearman = correlations[0]
    val_spearman = correlations[1]
    train_pearson = correlations[2]
    val_pearson = correlations[3]
    g.ax_joint.text(0.05, 0.95, f'Train Spearman: {train_spearman:.2f}', transform=g.ax_joint.transAxes, fontsize=12, verticalalignment='top', color='blue')
    g.ax_joint.text(0.05, 0.90, f'Train Pearson: {train_pearson:.2f}', transform=g.ax_joint.transAxes, fontsize=12, verticalalignment='top', color='blue')
    # g.ax_joint.text(0.05, 0.85, f'Train Correlation: {train_corr:.2f}', transform=g.ax_joint.transAxes, fontsize=12, verticalalignment='top', color='blue')
    # g.ax_joint.text(0.05, 0.90, f'Train mean: {statistics.mean(data[data["Type"] == "Training"]["Predicted"]):.2f}', transform=g.ax_joint.transAxes, fontsize=12, verticalalignment='top', color='blue')
    # g.ax_joint.text(0.05, 0.85, f'Train Std Dev: {statistics.stdev(data[data["Type"] == "Training"]["Predicted"]):.2f}', transform=g.ax_joint.transAxes, fontsize=12, verticalalignment='top', color='blue')
    g.ax_joint.text(0.05, 0.80, f'Val Spearman: {val_spearman:.2f}', transform=g.ax_joint.transAxes, fontsize=12, verticalalignment='top', color='orange')
    g.ax_joint.text(0.05, 0.75, f'Val Pearson: {val_pearson:.2f}', transform=g.ax_joint.transAxes, fontsize=12, verticalalignment='top', color='orange')
    # g.ax_joint.text(0.05, 0.70, f'Val Correlation: {val_corr:.2f}', transform=g.ax_joint.transAxes, fontsize=12, verticalalignment='top', color='orange')
    # g.ax_joint.text(0.05, 0.75, f'Val mean: {statistics.mean(data[data["Type"] == "Validation"]["Predicted"]):.2f}', transform=g.ax_joint.transAxes, fontsize=12, verticalalignment='top', color='orange')
    # g.ax_joint.text(0.05, 0.70, f'Val Std Dev: {statistics.stdev(data[data["Type"] == "Validation"]["Predicted"]):.2f}', transform=g.ax_joint.transAxes, fontsize=12, verticalalignment='top', color='orange')
    # g.ax_joint.text(0.05, 0.75, f'Val mean: {statistics.mean(data[data["Type"] == "Validation"]["Predicted"]):.2f}', transform=g.ax_joint.transAxes, fontsize=12, verticalalignment='top', color='orange')
    g.ax_joint.legend(title='Species Order', loc='lower right')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)     
    plt.savefig(output_dir + f"{species}-plot.png")
    plt.close()

def eda(input_file, output_dir):
    with h5py.File(input_file, 'r') as f:
        x = torch.tensor(f['x'][:], dtype=torch.float32)
        y = torch.tensor(f['y'][:], dtype=torch.float32)
        mask = torch.tensor(f['mask'][:], dtype=torch.float32)
        species = [s.decode('utf-8') for s in f['species_labels'][:]]
        protein = [p.decode('utf-8') for p in f['protein_labels'][:]]
        order = [o.decode('utf-8') for o in f['order_labels'][:]]
        # normalized_labels = torch.tensor(np.array([s.decode('utf-8') for s in f["normalized_labels"][:]], dtype=float))
        # standardized_labels = torch.tensor(np.array([s.decode('utf-8') for s in f["standardized_labels"][:]], dtype=float))
  
    prot_df = pd.DataFrame(protein, columns=[0])
    prot_df["species_id"] = prot_df[0].apply(lambda x: x.split(":")[0])
    prot_df["uniprot_id"] = prot_df[0].apply(lambda x: x.split(":")[1])
    prot_df.drop(columns=[0], inplace=True)
    order_df = pd.DataFrame()
    order_df["order"]  = order
    order_df["normalized_brain_size_residuals"] = y
    order_dict = order_df.set_index("order")["normalized_brain_size_residuals"].to_dict()
    for sp in prot_df["species_id"].unique():
        print(sp, len(prot_df[prot_df["species_id"] == sp]))
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].hist(y.numpy(), bins=30, alpha=0.7, color='blue')
    axs[0].set_title("Brain Residuals")
    axs[0].set_xlabel("Brain Residual")
    axs[0].set_ylabel("Count")
    order_freq = Counter(order)
    bars = axs[1].bar(list(order_freq.keys()), list(order_freq.values()), alpha=0.7, color='red')
    axs[1].set_title("Order Frequency with Average Normalized Brain Size Residuals")
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



def write_list_to_file(data, filename):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)


def min_max_normalize(lst):
    min_val = min(lst)
    max_val = max(lst)
    normalized_lst = [(x - min_val) / (max_val - min_val) for x in lst]
    return normalized_lst

def z_score_normalize(lst):
    mean_val = sum(lst) / len(lst)
    std_dev = (sum((x - mean_val) ** 2 for x in lst) / len(lst)) ** 0.5
    normalized_lst = [(x - mean_val) / std_dev for x in lst]
    return normalized_lst


def group_list(data_list, order_list):
    grouped_data = defaultdict(list)
    for item, order in zip(data_list, order_list):
        grouped_data[order].append(item)
    grouped_data = dict(grouped_data)
    return grouped_data

def fetch_uniprot_data(query, size=500):
    url = f"https://rest.uniprot.org/uniprotkb/search?query={query}&fields=accession,protein_name&format=tsv&size={size}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.text
        df = pd.read_csv(io.StringIO(data), sep='\t')
        return df
    else:
        print(f"Failed to retrieve data: {response.status_code}")
        return pd.DataFrame()

def get_all_uniprot_ids(query, size=500):
    all_data = pd.DataFrame()
    start = 0
    while True:
        url = f"https://rest.uniprot.org/uniprotkb/search?query={query}&fields=accession,protein_name&format=tsv&size={size}&offset={start}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.text
            df = pd.read_csv(io.StringIO(data), sep='\t')
            if df.empty or len(df) < size:
                all_data = pd.concat([all_data, df], ignore_index=True)
                break
            all_data = pd.concat([all_data, df], ignore_index=True)
            start += size
        else:
            print(f"Failed to retrieve data: {response.status_code}")
            break
    return all_data

def get_uniref50_id(uniprot_id):
    # url = f"https://rest.uniprot.org/uniref/UniRef90_{uniprot_id}.tsv"
    url = f"https://rest.uniprot.org/uniprotkb/search?query=accession_id:{uniprot_id}&identity:0.5&format=tsv"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.text
        df = pd.read_csv(io.StringIO(data), sep='\t')
        try:
            return df.iloc[0]['Cluster ID']
        except:
            pass
        
API_URL = "https://rest.uniprot.org"

def submit_id_mapping(from_db, to_db, ids):
    request = requests.post(
        f"{API_URL}/idmapping/run",
        data={"from": from_db, "to": to_db, "ids": ",".join(ids)},
    )
    request.raise_for_status()
    return request.json()["jobId"]

def check_id_mapping_results_ready(job_id):
    while True:
        request = requests.get(f"{API_URL}/idmapping/status/{job_id}")
        request.raise_for_status()
        j = request.json()
        if "jobStatus" in j:
            if j["jobStatus"] in ("NEW", "RUNNING"):
                print(f"Retrying in 3s")
                time.sleep(3)
            else:
                raise Exception(j["jobStatus"])
        else:
            return bool(j["results"] or j["failedIds"])

def get_id_mapping_results_link(job_id):
    url = f"{API_URL}/idmapping/details/{job_id}"
    request = requests.get(url)
    request.raise_for_status()
    return request.json()["redirectURL"]

def get_id_mapping_results(url):
    request = requests.get(url)
    request.raise_for_status()
    return request.json()
    
def map_uniprot_uniref(genes, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # uniprot api limited to 500 requesets
    if os.path.exists(f"{output_dir}/uniref.json"):
        with open(f"{output_dir}/uniref.json", "r") as f:
            data = json.load(f)
        finished_genes = flatten_list(list(data.values()))
        genes = list(set(genes) - set(finished_genes))
    if len(genes) >= 500: 
        batched_genes = [genes[i:i + 500] for i in range(0, len(genes), 500)]
    else: 
        batched_genes = [genes]
    uniref_members = {}
    
    for batch in batched_genes:
        job_id = submit_id_mapping(from_db="UniProtKB_AC-ID", to_db="UniRef50", ids=batch)
        if check_id_mapping_results_ready(job_id):
            link = get_id_mapping_results_link(job_id)
            results = get_id_mapping_results(link)
            uniref_members = {result['to']['id'] : result["to"]["representativeMember"]["accessions"] for result in results['results']}
            # uniref_members.update(tmp)
            update_json(f"{output_dir}uniref.json", uniref_members)
        
    print(f"Saved uniref dictionary to {output_dir}/uniref.json")


def update_json(file_to_update, data):
    if os.path.exists(file_to_update):
        with open(file_to_update, "r") as w:
            existing_data = json.load(w)
            existing_data.update(data)
        with open(file_to_update, "w") as w:
            json.dump(existing_data, w)
    else:
        with open(file_to_update, "w") as w:
            json.dump(data, w)

    # one hot
    # metadata = pd.read_csv("/home/gluetown/brain/data/metadata_all_2.csv.gz", compression = "gzip")
    # onehot = {species: [] for species in metadata["species_id"].unique()}
    # for species in list(onehot.keys()):
    #     tmp = metadata.loc[metadata["species_id"] == species]
    #     onehot[species] = [1 if i in all_genes else 0 for i in tmp["uniprot_id"]]
    # with open(f"{output_dir}/onehot.json", "w") as f:
    #     json.dump(onehot, f)
    # print(f"Saved onehot to {output_dir}/onehot.json")
    # num_intersection = set(all_genes).intersection(set(metadata["uniprot_id"]))
    # print(f"{len(num_intersection)} genes in dataset")

def write_output(output_dir, fold_actuals, fold_preds, fold_train_actuals, fold_train_preds, fold_train_losses, fold_val_losses, crit_indices, spearman_baseline_correlation,  pearson_baseline_correlation, x_list, y_list):
    with open(output_dir + "pearson_baseline_correalation.txt", "w") as w:
        json.dump(pearson_baseline_correlation, w, indent=4)
    with open(output_dir + "spearman_baseline_correalation.txt", "w") as w:
        json.dump(spearman_baseline_correlation, w, indent=4)
    write_list_to_file(fold_actuals, output_dir + "fold_actuals.csv")
    write_list_to_file(fold_preds, output_dir + "fold_preds.csv")
    write_list_to_file(fold_train_actuals, output_dir + "fold_train_actuals.csv")
    write_list_to_file(fold_train_preds, output_dir + "fold_train_preds.csv")
    write_list_to_file(fold_train_losses, output_dir + "fold_train_losses.csv")
    write_list_to_file(fold_val_losses, output_dir + "fold_val_losses.csv")  
    with open(output_dir + "crit_idx.json", "w") as w:
        json.dump(crit_indices, w) 
    # eda(input_file, output_dir)
    shutil.copy(__file__, output_dir + os.path.basename(__file__))
    all_correlation = {"train_spearman": [compute_spearman_correlation(i, j) for i, j in zip(fold_train_actuals, fold_train_preds)], 
                        "val_spearman": [compute_spearman_correlation(i, j) for i, j in zip(fold_actuals, fold_preds)], 
                        "train_pearson":  [compute_pearson_correlation(i, j) for i, j in zip(fold_train_actuals, fold_train_preds)],  
                        "val_pearson": [compute_pearson_correlation(i, j) for i, j in zip(fold_actuals, fold_preds)], 
                        "base_spearman": compute_spearman_correlation([i.shape[1] for i in x_list], y_list),
                        "base_pearson": compute_pearson_correlation([i.shape[1] for i in x_list], y_list)}
    with open(output_dir + "all_correlation.json", "w") as w:
        json.dump(all_correlation, w)

def forward_wrapper(inputs, model):
    int_inputs = inputs.long()  # Convert back to integer type
    outputs = model(int_inputs)
    return outputs.last_hidden_state  # Return the last hidden state

def process_fasta(input_file, output_file):
    my_dict = SeqIO.to_dict(SeqIO.parse(input_file, "fasta"))
    processed_dict = {k.split("|")[1]: str(v.seq) for k,v in my_dict.items()}
    update_json(output_file, processed_dict)
    print(f"{input_file} dictionary saved to {output_file}")


############### INITIALIZATION ####################

### TEST SET ###
# seed = 42 
# set_seed(seed)
# input_file = "/home/gluetown/brain/data/embeddings/test/go_terms/20.h5"
# pad_x, y, order_labels, vocab, mask = load_output(input_file, order=True)
# mask = mask.unsqueeze(2)
# log_memory_usage()
# input_dim = pad_x.shape[1]
# output_dim = 1
# batch_size = 10
# learning_rate = 1e-3
# num_folds = 3
# max_epochs = 10
# early_stopping = True
# output_dir = f"/home/gluetown/brain/data/embeddings/test/point_net_masked_{input_file.split('/')[-1].split('.')[0]}_max_epochs{max_epochs}/"
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)
# model, fold_actuals, fold_preds, fold_train_actuals, fold_train_preds, fold_train_losses, fold_val_losses, train_spearmans, val_spearmans = train_and_evaluate(pad_x, y, order_labels, mask, input_dim, output_dim,  batch_size, learning_rate, num_folds, max_epochs, early_stopping, output_dir)
# plot_scatter(fold_actuals, fold_preds, fold_train_actuals, fold_train_preds, train_spearmans, val_spearmans, output_dir)
# number_of_genes = mask.sum(dim=1)
# compute_spearman_correlation(number_of_genes, y) # 0.13443633565768787

# # Compare masked vs unmasked
# output_dir = f"/home/gluetown/brain/data/embeddings/test/point_net_unmasked_{input_file.split('/')[-1].split('.')[0]}_max_epochs{max_epochs}/"
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)
# mask=None
# model, fold_actuals, fold_preds, fold_train_actuals, fold_train_preds, fold_train_losses, fold_val_losses, train_spearmans, val_spearmans = train_and_evaluate(pad_x, y, order_labels, mask, input_dim, output_dim,  batch_size, learning_rate, num_folds, max_epochs, early_stopping, output_dir)
# plot_scatter(fold_actuals, fold_preds, fold_train_actuals, fold_train_preds, train_spearmans, val_spearmans, output_dir)

### GYS DATASET ### 
if __name__ == "__main__": 
    input_file = "/group/gquongrp/collaborations/brain/embeddings.pt"
    x_dict = torch.load(input_file)
    max_len = max(tensor.shape[0] for tensor in x_dict.values())
    # pad_x = torch.zeros(len(x_dict.keys()), max_len, 320)
    # mask = torch.zeros(len(x_dict.keys()), max_len)
    species_labels = []
    for i, (species, tensor) in enumerate(x_dict.items()):
        species_labels.append(species)
                    # pad_x[i, :tensor.shape[0], :] = tensor
                    # mask[i, :tensor.shape[0]] = 1
                    # if len(mask.shape) != 2: 
                    #     mask = mask.unsqueeze(2)
    labels_file = "/group/gquongrp/collaborations/brain/labels.pt"
    y_dict = torch.load(labels_file)
    # y = [y_dict[k] for k in species_labels]
    # y = torch.tensor(y)
    group_file = "/group/gquongrp/collaborations/brain/common_names.csv"
    group_dict = pd.read_csv(group_file).set_index("Proteome_ID")["order"].to_dict()
    order_dict = {k.split("_")[0]:group_dict[k.split("_")[0]] for k in species_labels}
    order_labels = [group_dict[k.split("_")[0]] for k in species_labels]
    vocab = {label: i for i, label in enumerate(set(order_labels))}
    order_labels = [vocab[label] for label in order_labels]
    key_mapping = {i:i.split("_")[0] for i in x_dict.keys()}
    x_dict = {key_mapping.get(k, k): v.float() for k, v in x_dict.items()}
    y_dict = {key_mapping.get(k, k): v.astype(np.float32) for k, v in y_dict.items()}
    x_list = [x_dict[i].permute(1,0) for i in sorted(list(x_dict.keys()))]
    y_list = [y_dict[i] for i in sorted(list(x_dict.keys()))]
    y_list_normalized = min_max_normalize(y_list)
    y_list_standardized = z_score_normalize(y_list)
    # combines orders with > 1 species
    all_orders = [order_dict[i] for i in sorted(list(x_dict.keys()))]
    order_list = [i if Counter(all_orders)[i] > 5 else "Other" for i in all_orders]
    log_memory_usage()
    seed = 42 
    set_seed(seed)
    input_dim = 320
    output_dim = 1
    batch_size = 5
    learning_rate = 1e-4
    num_folds = 5
    max_epochs = 50
    early_stopping = False
    output_dir = f"/home/gluetown/brain/data/embeddings/test/gys/point_net_list_gys_logo_max_epochs{max_epochs}_normalized/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model, fold_actuals, fold_preds, fold_train_actuals, fold_train_preds, fold_train_losses, fold_val_losses, train_spearmans, val_spearmans = train_and_evaluate_logo(x_list, y_list_normalized, order_list, list(set(order_list)), input_dim, output_dim,  batch_size, learning_rate, num_folds, max_epochs, early_stopping, output_dir)
    write_list_to_file(fold_actuals, output_dir + "fold_actuals.csv")
    write_list_to_file(fold_preds, output_dir + "fold_preds.csv")
    write_list_to_file(fold_train_actuals, output_dir + "fold_train_actuals.csv")
    write_list_to_file(fold_train_preds, output_dir + "fold_train_preds.csv")
    write_list_to_file(fold_train_losses, output_dir + "fold_train_losses.csv")
    write_list_to_file(fold_val_losses, output_dir + "fold_val_losses.csv")
    shutil.copy(__file__, output_dir + os.path.basename(__file__))  

### (GYS, DRYAD) DATASET ### 
if __name__ == "__main__":
    dryad_file = "/home/gluetown/brain/data/embeddings/h5_files/dryad_reg_out_order.h5"
    pad_x1, y1, order_labels1, vocab1, mask1, species_labels1, protein_labels1  = load_output(dryad_file, protein_labels = True, order = True, indiv_datasets = True)
    x_list1, y_list1, y_list_normalized1, y_list_standardized1, order_list1, mask1, species_labels1 =  initialize_dataset(pad_x1, y1, order_labels1, vocab1, mask1, species_labels1)
    gys_file = "/home/gluetown/brain/data/embeddings/h5_files/gys_reg_out_order.h5"
    pad_x2, y2, order_labels2, vocab2, mask2, species_labels2, protein_labels2  = load_output(gys_file, protein_labels = True, order = True, indiv_datasets = True)        
    x_list2, y_list2, y_list_normalized2, y_list_standardized2, order_list2, mask2, species_labels2 =  initialize_dataset(pad_x2, y2, order_labels2, vocab2, mask2, species_labels2)
    x_list = x_list1 + x_list2
    y_list_normalized = y_list_normalized1 + y_list_normalized2
    order_list = order_list1 + order_list2
    species_labels = species_labels1 + species_labels2
    protein_labels = protein_labels1.update(protein_labels2)
    x_group = group_list([i.shape[1] for i in x_list], order_list)
    y_group = group_list(np.array(y_list_normalized), order_list)
    pearson_baseline_correlation = {sp: compute_pearson_correlation(x_group[sp], y_group[sp]) for sp, sp_y in zip(x_group,y_group) }
    spearman_baseline_correlation = {sp: compute_spearman_correlation(x_group[sp], y_group[sp]) for sp, sp_y in zip(x_group,y_group) }
    seed = 42
    set_seed(seed)
    log_memory_usage()
    input_dim = x_list[0].shape[0]
    output_dim = 1
    batch_size = 10
    learning_rate = 1e-4
    num_folds = 10
    max_epochs = 50
    use_early_stopping = True
    hidden_dim = 256
    use_dropout = True
    num_layers = 2

    output_dir = f"/home/gluetown/brain/data/embeddings/test/full/gys_dryad_point_net_encoder_not_group_kfold_epoch_{max_epochs}_num_layers{num_layers}_hidden_dim_{hidden_dim}_seed{seed}"
    if use_early_stopping: output_dir += "_early_stopping"
    if use_dropout: output_dir += "_dropout"    
    output_dir += "/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model, fold_actuals, fold_preds, fold_train_actuals, fold_train_preds, fold_train_losses, fold_val_losses, train_spearmans, val_spearmans,  train_num_genes, all_train_labels, val_num_genes, all_val_labels, crit_indices = train_and_evaluate_kfold(x_list, y_list_normalized, order_list, species_labels, input_dim, output_dim, hidden_dim, batch_size, learning_rate, num_folds, max_epochs, use_early_stopping, use_dropout, num_layers, output_dir)
    write_output(output_dir, fold_actuals, fold_preds, fold_train_actuals, fold_train_preds, fold_train_losses, fold_val_losses, crit_indices, spearman_baseline_correlation,  pearson_baseline_correlation, x_list, y_list_normalized)
    # eda(input_file, output_dir)
    shutil.copy(__file__, output_dir + os.path.basename(__file__))

# train_pearsons = []
# val_pearsons = []
# train_spearmans = []
# val_spearmans = []
# input_dir = "/home/gluetown/brain/data/embeddings/test/full/gys_dryad_point_net_not_group_kfold_normalized_labels_epoch_50_num_layers2_hidden_dim_256_seed42_early_stopping_dropout/"
# train_actual = pd.read_csv(input_dir + "fold_train_actuals.csv")
# train_preds = pd.read_csv(input_dir + "fold_train_preds.csv")
# val_actual = pd.read_csv(input_dir + "fold_actuals.csv")
# val_pred = pd.read_csv(input_dir + "fold_preds.csv")
# train_spearmans.append(compute_spearman_correlation(train_actual, train_preds))
# val_spearmans.append(compute_spearman_correlation(val_actual, val_pred))
# train_pearsons.append(compute_pearson_correlation(train_actual, train_preds.flatten())) 
# val_pearsons.append(compute_pearson_correlation(val_actual, val_pred.flatten()))
# correlations = [compute_spearman_correlation(train_actual, train_preds), compute_spearman_correlation(val_actual, val_pred), compute_pearson_correlation(train_actual, train_preds.flatten()), compute_pearson_correlation(val_actual, val_pred.flatten())]


# #### Dryad Dataset ### 
# # if __name__ == "__main__": 
# #     dryad_file = "/home/gluetown/brain/data/embeddings/h5_files/dryad.h5"
# #     pad_x, y, order_labels, vocab, mask, species_labels  = load_output(dryad_file, order=True)
# #     x_list = []
# #     for x in remove_mask(pad_x, mask):
# #         x_list.append(x.permute(1,0))    
# #     y_list = list(y)
# #     y_list_normalized = min_max_normalize(y_list)
# #     y_list_standardized = z_score_normalize(y_list)
# #     swapped = {v:k for k, v in vocab.items()}
# #     all_orders = [swapped[k] for k in order_labels]
# #     # temporary solution to species - embeddings mismatch: 
# #     trouble_makers = ["UP000189704_1868482.fasta", "UP000009136_9913.fasta", "UP000694520_30521.fasta"]
# #     troublemaker_indices = list(np.array([get_indices(i, species_labels) for i in trouble_makers]).flatten())
# #     x_list = [x_list[i] for i in range(len(x_list)) if i not in troublemaker_indices]
# #     order_list = [i if Counter(all_orders)[i] > 5 else "Other" for i in all_orders]
# #     seed = 30
# #     set_seed(seed)
# #     log_memory_usage()
# #     input_dim = x_list[0].shape[0]
# #     output_dim = 1
# #     batch_size = 10
# #     learning_rate = 1e-4
# #     num_folds = 6
# #     max_epochs = 50
# #     use_early_stopping = False
# #     hidden_dim = 512
# #     use_dropout = False
# #     output_dir = f"/home/gluetown/brain/data/embeddings/test/full/gys_tsuboi_point_net_list_full_kfold_normalized_labels_epoch_{max_epochs}_early_stopping_patience_30_delta_0.01_batchnorm_256_backbone_hidden_dim_{hidden_dim}_dropout_seed30/"
# #     if not os.path.exists(output_dir):
# #         os.makedirs(output_dir)
# #     model, fold_actuals, fold_preds, fold_train_actuals, fold_train_preds, fold_train_losses, fold_val_losses, train_spearmans, val_spearmans,  train_num_genes, all_train_labels, val_num_genes, all_val_labels = train_and_evaluate_kfold(x_list, y_list_normalized, order_list, list(set(order_list)), input_dim, output_dim, hidden_dim, batch_size, learning_rate, num_folds, max_epochs, use_early_stopping, use_dropout, output_dir)
# #     # train_baseline_correlation = [compute_pearson_correlation(t, l) for t, l in zip(train_num_genes, all_train_labels)]
# #     # val_baseline_correlation = [compute_pearson_correlation(t, l) for t, l in  zip(val_num_genes, all_val_labels)]
# #     write_list_to_file(fold_actuals, output_dir + "fold_actuals.csv")
# #     write_list_to_file(fold_preds, output_dir + "fold_preds.csv")
# #     write_list_to_file(fold_train_actuals, output_dir + "fold_train_actuals.csv")
# #     write_list_to_file(fold_train_preds, output_dir + "fold_train_preds.csv")
# #     write_list_to_file(fold_train_losses, output_dir + "fold_train_losses.csv")
# #     write_list_to_file(fold_val_losses, output_dir + "fold_val_losses.csv")
# #     write_list_to_file(train_num_genes, output_dir + "train_num_genes.csv")
# #     write_list_to_file(all_train_labels, output_dir + "all_train_labels.csv")
# #     write_list_to_file(val_num_genes, output_dir + "val_num_genes.csv")
# #     write_list_to_file(all_val_labels, output_dir + "all_val_labels.csv")
# #     # write_list_to_file(train_baseline_correlation, output_dir + "train_baseline_correlation.csv")
# #     # write_list_to_file(val_baseline_correlation, output_dir + "val_baseline_correlation.csv")
# #     # eda(input_file, output_dir)
# #     shutil.copy(__file__, output_dir + os.path.basename(__file__))
    
# # # FULL DATASET (GYS, TSUBOI, DRYAD) #### 
# # if __name__ == "__main__": 
# #     input_file = "/home/gluetown/brain/data/embeddings/h5_files/all_3.h5"
# #     pad_x, y, order_labels, vocab, mask, species_labels  = load_output(input_file, order=True)
# #     x_list, y_list, y_list_normalized, y_list_standardized, order_list, mask, species_labels =  initialize_dataset(pad_x, y, order_labels, vocab, mask, species_labels)
# #     x_group = group_list([i.shape[1] for i in x_list], order_list)
# #     y_group = group_list(np.array(y_list), order_list)
# #     pearson_baseline_correlation = {sp: compute_pearson_correlation(x_group[sp], y_group[sp]) for sp, sp_y in zip(x_group,y_group) }
# #     spearman_baseline_correlation = {sp: compute_spearman_correlation(x_group[sp], y_group[sp]) for sp, sp_y in zip(x_group,y_group) }
# #     seed = 42
# #     set_seed(seed)
# #     log_memory_usage()
# #     input_dim = x_list[0].shape[0]
# #     output_dim = 1
# #     batch_size = 10
# #     learning_rate = 1e-4
# #     num_folds = 10
# #     max_epochs = 50
# #     use_early_stopping = False
# #     hidden_dim = 256
# #     use_dropout = False
# #     output_dir = f"/home/gluetown/brain/data/embeddings/test/full/point_net_not_group_kfold_epoch_{max_epochs}_batchnorm_256_backbone_hidden_dim_{hidden_dim}_seed{seed}"
# #     if use_early_stopping: output_dir += "_early_stopping"
# #     if use_dropout: output_dir += "_dropout"    
# #     output_dir += "/"
# #     if not os.path.exists(output_dir):
# #         os.makedirs(output_dir)
# #     model, fold_actuals, fold_preds, fold_train_actuals, fold_train_preds, fold_train_losses, fold_val_losses, train_spearmans, val_spearmans,  train_num_genes, all_train_labels, val_num_genes, all_val_labels = train_and_evaluate_kfold(x_list, y_list_normalized, order_list,species_labels, input_dim, output_dim, hidden_dim, batch_size, learning_rate, num_folds, max_epochs, use_early_stopping, use_dropout, output_dir)
# #     write_output(output_dir, fold_actuals, fold_preds, fold_train_actuals, fold_train_preds, fold_train_losses, fold_val_losses,spearman_baseline_correlation,  pearson_baseline_correlation, x_list, y_list_normalized)
    

           

# #### Running each dataset independently ###
# # if __name__ == "__main__":
# #     datasets = ["gys", "tsuboi", "dryad"]

# #     for dataset in datasets:
# #         replace_y_file = f"/home/gluetown/brain/data/embeddings/h5_files/{dataset}_reg_out_order_residuals.pkl"
# #         with open(replace_y_file, 'rb') as file:
# #             replacement_y = pkl.load(file)
# #         h5_file = f"/home/gluetown/brain/data/embeddings/h5_files/{dataset}.h5"
# #         pad_x, y, order_labels, vocab, mask, species_labels  = load_output(h5_file, order=True)
# #         x_list, y_list, y_list_normalized, y_list_standardized, order_list, mask, species_labels =  initialize_dataset(pad_x, y, order_labels, vocab, mask, species_labels)
# #         y_list = [replacement_y[spe] for spe in species_labels]
# #         y_list_normalized = min_max_normalize(y_list)
# #         x_group = group_list([i.shape[1] for i in x_list], order_list)
# #         y_group = group_list(np.array(y_list), order_list)
# #         pearson_baseline_correlation = {sp: compute_pearson_correlation(x_group[sp], y_group[sp]) for sp, sp_y in zip(x_group,y_group) }
# #         spearman_baseline_correlation = {sp: compute_spearman_correlation(x_group[sp], y_group[sp]) for sp, sp_y in zip(x_group,y_group) }

# #         # initialization
# #         seed = 42
# #         set_seed(seed)
# #         log_memory_usage()
# #         input_dim = x_list[0].shape[0]
# #         output_dim = 1
# #         batch_size = 10
# #         learning_rate = 1e-4
# #         num_folds = len(set(order_list))
# #         max_epochs = 50
# #         use_early_stopping = True
# #         hidden_dim = 256
# #         use_dropout = True
# #         output_dir = f"/home/gluetown/brain/data/embeddings/test/full/{dataset}_point_net_not_group_kfold_epoch_{max_epochs}_batchnorm_256_backbone_hidden_dim_{hidden_dim}_seed{seed}"
# #         if use_early_stopping: output_dir += "_early_stopping"
# #         if use_dropout: output_dir += "_dropout"
# #         output_dir += "/"
# #         if not os.path.exists(output_dir):
# #             os.makedirs(output_dir)
# #         model, fold_actuals, fold_preds, fold_train_actuals, fold_train_preds, fold_train_losses, fold_val_losses, train_spearmans, val_spearmans,  train_num_genes, all_train_labels, val_num_genes, all_val_labels, crit_indices = train_and_evaluate_kfold(x_list, y_list_normalized, order_list, list(set(order_list)), input_dim, output_dim, hidden_dim, batch_size, learning_rate, num_folds, max_epochs, use_early_stopping, use_dropout, output_dir)

# #         write_output(output_dir, fold_actuals, fold_preds, fold_train_actuals, fold_train_preds, fold_train_losses, fold_val_losses, crit_indices, spearman_baseline_correlation,  pearson_baseline_correlation, x_list, y_list_normalized)
    

# ############ EDA #################
# # output_dir = "/home/gluetown/brain/data/embeddings/test/gys/eda/"
# # y_list = list(y)
# # y_list_normalized = min_max_normalize(y_list)
# # y_list_normalized_tensor = torch.tensor(y_list_normalized, dtype=torch.float32)
# # y_list_z_normalized = z_score_normalize(y_list)
# # y_list_z_normalized_tensor = torch.tensor(y_list_z_normalized, dtype=torch.float32)
# # plt.figure(figsize=(12, 6))
# # plt.hist(y_list, bins=30, alpha=0.5, label='Raw residuals', color='gold')
# # plt.hist(y_list_normalized, bins=30, alpha=0.5, label='Normalized residuals', color='blue')
# # plt.hist(y_list_z_normalized, bins=30, alpha=0.5, label='Standardized residuals', color='pink')
# # plt.title('Histograms of Normalized and Standardized Residuals Gys Dataset (109 species)')
# # plt.xlabel('Value')
# # plt.ylabel('Frequency')
# # plt.legend()
# # plt.show()
# # plt.savefig(output_dir + "gys_dist_of_residuals.png")

# ### Plot num_genes vs residuals ###
# # if __name__ == "__main__":
# #     datasets = ["gys", "tsuboi", "dryad"]
# #     for dataset in datasets:
# #         output_dir = "/home/gluetown/brain/data/embeddings/test"
# #         output_dir += dataset + "/"
# #         if not os.path.exists(output_dir):
# #             os.makedirs(output_dir)
# #         replace_y_file = f"/home/gluetown/brain/data/embeddings/h5_files/{dataset}_reg_out_order_residuals.pkl"
# #         with open(replace_y_file, 'rb') as file:
# #             replacement_y = pkl.load(file)
# #         h5_file = f"/home/gluetown/brain/data/embeddings/h5_files/{dataset}.h5"
# #         pad_x, y, order_labels, vocab, mask, species_labels  = load_output(h5_file, order=True)
# #         x_list, y_list, y_list_normalized, y_list_standardized, order_list, mask, species_labels =  initialize_dataset(pad_x, y, order_labels, vocab, mask, species_labels)
# #         y_list = [replacement_y[spe] for spe in species_labels]
# #         y_list_normalized = min_max_normalize(y_list)
# #         x_group = group_list([i.shape[1] for i in x_list], order_list)
# #         y_group = group_list(np.array(y_list_normalized), order_list)
# #         pearson_baseline_correlation = {sp: compute_pearson_correlation(x_group[sp], y_group[sp]) for sp, sp_y in zip(x_group,y_group) }
# #         spearman_baseline_correlation = {sp: compute_spearman_correlation(x_group[sp], y_group[sp]) for sp, sp_y in zip(x_group,y_group) }
# #         avg_num_genes = {k:mean(v) for k,v in x_group.items()}
# #         avg_residuals = {k:mean(v) for k,v in y_group.items()}
# #         num_genes = [i.shape[1] for i in x_list] 
# #         df = pd.DataFrame({
# #             'num_genes': num_genes,
# #             'residuals': np.array(y_list_normalized),
# #             'order': order_list
# #         })
# #         corr_df = pd.DataFrame({
# #             "pearson": [pearson_baseline_correlation[k] for k in  sorted(list(pearson_baseline_correlation.keys()))], 
# #             "spearman": [spearman_baseline_correlation[k] for k in  sorted(list(spearman_baseline_correlation.keys()))], 
# #             "avg_num_genes": [avg_num_genes[k] for k in  sorted(list(avg_num_genes.keys()))], 
# #             "order": sorted(list(avg_num_genes.keys())), 
# #             "avg_residuals": [avg_residuals[k] for k in  sorted(list(avg_residuals.keys()))]
# #         })
# #         unique_orders = corr_df['order'].unique()
# #         custom_palette = sns.color_palette("hsv", len(unique_orders))
# #         palette_dict = {order: color for order, color in zip(unique_orders, custom_palette)}
# #         # Plot the pearson plot
# #         plt.figure(figsize=(20, 10))
# #         scatter = sns.scatterplot(data=corr_df, x='avg_residuals', y='pearson', hue='order', palette=palette_dict)
# #         sns.scatterplot(data=corr_df, x='avg_residuals', y='pearson', hue='order', palette=palette_dict)
# #         plt.title('Scatter Plot of avg_residuals vs pearson')
# #         plt.xlabel('avg_residuals')
# #         plt.ylabel('pearson')
# #         handles, labels = scatter.get_legend_handles_labels()
# #         new_labels = []
# #         for label in labels:
# #             if label in spearman_baseline_correlation and label in avg_num_genes:
# #                 new_label = f"{label} (spearman: {spearman_baseline_correlation[label]:.2f}, avg_num_genes: {avg_num_genes[label]:.2f})"
# #                 new_labels.append(new_label)
# #             else:
# #                 new_labels.append(label)
                
# #         scatter.legend(handles=handles, labels=new_labels, title='Order')
# #         plt.savefig(f"{output_dir}/pearson_corr_residuals.png")
# #         plt.show()

# #         # Plot the spearman plot
# #         plt.figure(figsize=(20, 10))
# #         scatter = sns.scatterplot(data=corr_df, x='avg_residuals', y='spearman', hue='order', palette=palette_dict)
# #         sns.scatterplot(data=corr_df, x='avg_residuals', y='spearman', hue='order', palette=palette_dict)
# #         plt.title('Scatter Plot of avg_residuals vs spearman')
# #         plt.xlabel('avg_residuals')
# #         plt.ylabel('spearman')
# #         handles, labels = scatter.get_legend_handles_labels()
# #         new_labels = []
# #         for label in labels:
# #             if label in pearson_baseline_correlation and label in avg_num_genes:
# #                 new_label = f"{label} (Pearson: {pearson_baseline_correlation[label]:.2f}, avg_num_genes: {avg_num_genes[label]:.2f})"
# #                 new_labels.append(new_label)
# #             else:
# #                 new_labels.append(label) 

# #         scatter.legend(handles=handles, labels=new_labels, title='Order')
# #         plt.savefig(f"{output_dir}/spearman_corr_residuals.png")
# #         plt.show()

# #         # Plot the num_genes_residuals plot
# #         plt.figure(figsize=(20, 10))
# #         scatter = sns.scatterplot(data=df, x='num_genes', y='residuals', hue='order', palette=palette_dict)
# #         sns.scatterplot(data=df, x='num_genes', y='residuals', hue='order', palette=palette_dict)
# #         plt.title('Scatter Plot of num_genes vs residuals')
# #         plt.xlabel('Number of Genes')
# #         plt.ylabel('Residuals')
# #         handles, labels = scatter.get_legend_handles_labels()
# #         new_labels = []
# #         for label in labels:
# #             if label in pearson_baseline_correlation and label in spearman_baseline_correlation:
# #                 new_label = f"{label} (Pearson: {pearson_baseline_correlation[label]:.2f}, Spearman: {spearman_baseline_correlation[label]:.2f})"
# #                 new_labels.append(new_label)
# #             else:
# #                 new_labels.append(label)

# #         scatter.legend(handles=handles, labels=new_labels, title='Order')
# #         plt.savefig(f"{output_dir}/num_genes_residuals.png")
# #         plt.show()



# ############ WEAKLY SUPERVISED LEARNING ###############
# # get microcephaly genes
# # if __name__ == "__main__":
# #     #     # positives
# #     #     # getting microcephaly genes
# #     #     # microcephaly_genes  = list(pd.read_csv("/home/gluetown/brain/data/weak_sup/uniprotkb_microcephaly_2024_11_13.tsv.gz", delimiter = "\t", compression = "gzip")["Entry"])
# #     #     brain_genes = list(pd.read_csv("/home/gluetown/brain/data/weak_sup/uniprotkb_brain_2024_11_13.tsv.gz", delimiter = "\t", compression = "gzip")["Entry"])
# #     #     print(f"{len(brain_genes)} brain genes")
# #     #     batched_brain_genes = [brain_genes[i:i + 500] for i in range(0, len(brain_genes), 500)]
# #         # negatives
# #         # organ_folder = "/home/gluetown/brain/data/weak_sup/organs/"
# #         # for file in os.listdir(organ_folder):
# #         #     organ_genes = list(pd.read_csv(organ_folder + file, delimiter = "\t", compression = "gzip")["Entry"])
# #         #     map_uniprot_uniref(organ_genes, "/home/gluetown/brain/data/" + file.split("_")[1])

# #     input_file = "/home/gluetown/brain/data/embeddings/h5_files/all_3.h5"
# #     pad_x, y, order_labels, vocab, mask, species_labels, protein_labels  = load_output(input_file, protein_labels = True, order = True, indiv_datasets = False)
# #     # pad_x, y, order_labels, vocab, mask, species_labels, protein_labels  = load_output(input_file, protein_labels = True, order = True)
# #     x_list, y_list, y_list_normalized, y_list_standardized, order_list, mask, species_labels =  initialize_dataset(pad_x, y, order_labels, vocab, mask, species_labels)
# #     protein_label_df = pd.DataFrame([item.split(':') for item in protein_labels], columns=['species_id', 'uniprot_id'])
# #     protein_label_dict = {}
# #     for species_id in protein_label_df["species_id"].unique():
# #         tmp = protein_label_df.loc[protein_label_df['species_id'] == species_id]
# #         protein_label_dict[tmp["species_id"].iloc[0]] = list(tmp["uniprot_id"].values)
            
# #     organ_folder = "/home/gluetown/brain/data/weak_sup/uniprot_files/"
# #     all_genes = set(list(pd.read_csv("/home/gluetown/brain/data/weak_sup/all_genes.csv", header=None)[0].values))


# #     for file in os.listdir(organ_folder):
# #         organ = file.split("_")[1]
# #         organ_genes = list(pd.read_csv(organ_folder + file, compression = "gzip", delimiter = "\t")["Entry"].values)
# #         organ_genes = set(organ_genes).intersection(all_genes)
# #         organ_genes_df = pd.DataFrame(list(organ_genes), columns=["Gene"])
# #         organ_genes_df.to_csv(f"/home/gluetown/brain/data/weak_sup/negative/{organ}_genes.csv", index=False)

# #     # get all organ genes (annotated)
# #     all_organ_genes = []
# #     saved_folder = "/home/gluetown/brain/data/weak_sup/negative/"
# #     for file in os.listdir(saved_folder):
# #         if file.endswith(".csv"):
# #             all_organ_genes += list(pd.read_csv(saved_folder + file)["Gene"].values)
# #     # extract annotated genes into new x_list, 
# #     # y = onehot of brain / other organ annotation, 
# #     # species labels
# #     # protein labels
# #     species_gene_indices = {}
# #     for s in species_labels: 
# #         species_gene_indices[s] = flatten_list([np.where(i in all_organ_genes)[0].tolist() for i in protein_label_dict[s]])
        
# #     filtered_x_list = [np.take(x_list[i], species_gene_indices[species_labels[i]]) for i in range(len(x_list))]    
# #     filtered_protein_list = [np.take(protein_label_dict[s], species_gene_indices[species_labels[s]]) for s in species_labels]    
# #     positive = list(pd.read_csv("/home/gluetown/brain/data/weak_sup/positive/brain_genes.csv", header = None)[0].values)
# #     onehot = {}
# #     for ind, species in enumerate(species_labels):
# #         onehot[species] = np.put(np.zeros(filtered_x_list[ind].shape[1]), species_gene_indices[species], 1)
        
# #     output_file = "/home/gluetown/brain/data/weak_sup/filtered.h5"
# #     with h5py.File(output_file, "w") as f:
# #         f.create_dataset("x", data=filtered_x_list)
# #         f.create_dataset("y", data=y_list_normalized)
# #         f.create_dataset("protein_labels", data=np.array(filtered_protein_list, dtype='S'))
# #         f.create_dataset("species_labels", data=np.array(species_labels, dtype='S'))
# #         f.create_dataset("order_labels", data=np.array(order_labels, dtype='S'))
        


# # with open("/home/gluetown/brain/data/weak_sup/brain_uniref.json", "r") as f:
# #     uniref_members = json.load(f)
    
# # with open("/home/gluetown/brain/data/weak_sup/brain_onehot.json", "r") as f:
# #     onehot = json.load(f)

# # weakly supervised model
# # dataloader: list (length s = num_species) of gene matrices (nx320)
# # input: gene matrix (nx320)
#     # input_dim = 320
# # output: binary array (one hot of microcephaly genes)

# # class PointNetRegHead(nn.Module):
# #     def __init__(self, input_dim, output_dim, hidden_dim, num_layers):
# #         super(PointNetRegHead, self).__init__()
# #         self.linear = nn.Linear(input_dim, 256)
# #         layers = []
# #         for _ in range(num_layers):
# #             layers.append(nn.Linear(256, 256))
# #             layers.append(nn.ReLU())
# #         self.hidden_layers = nn.Sequential(*layers)
# #         self.out = nn.Linear(256, output_dim)
# #     def forward(self, x):
        
# # organ_file = "/home/gluetown/brain/data/weak_sup/filtered.h5"
# # pad_x, y, order_labels, vocab, mask, species_labels, protein_labels  = load_output(organ_file, protein_labels = True, order = True, indiv_datasets = False)

# ########## UMAP #### ### 
# # if __name__ == "__main__":     
#     # with open("/home/gluetown/brain/data/weak_sup/positive/brain_uniref.json", "r") as f:
#     #     brain_uniref = json.load(f)

#     # all_brain_genes = list(pd.read_csv("/home/gluetown/brain/data/weak_sup/overlap_brain_genes.csv", header=None)[0].values)
#     # dryad_file = "/home/gluetown/brain/data/embeddings/h5_files/dryad_reg_out_order.h5"
#     # pad_x1, y1, order_labels1, vocab1, mask1, species_labels1, protein_labels1  = load_output(dryad_file, protein_labels = True, order = True)
#     # x_list1, y_list1, y_list_normalized1, y_list_standardized1, order_list1, mask1, species_labels1 =  initialize_dataset(pad_x1, y1, order_labels1, vocab1, mask1, species_labels1)
#     # gys_file = "/home/gluetown/brain/data/embeddings/h5_files/dryad_reg_out_order.h5"
#     # pad_x2, y2, order_labels2, vocab2, mask2, species_labels2, protein_labels2  = load_output(gys_file, protein_labels = True, order = True)   
#     # x_list2, y_list2, y_list_normalized2, y_list_standardized2, order_list2, mask2, species_labels2 =  initialize_dataset(pad_x2, y2, order_labels2, vocab2, mask2, species_labels2)
#     # x_list = x_list1 + x_list2
#     # y_list_normalized = y_list_normalized1 + y_list_normalized2
#     # order_list = order_list1 + order_list2
#     # species_labels = species_labels1 + species_labels2
#     # protein_labels1.update(protein_labels2) 
#     # data = [i.permute(1, 0) for i in x_list]
#     # protein_label_df = pd.DataFrame([item.split(':') for item in protein_labels], columns=['species_id', 'uniprot_id'])





#     # # for this model, only gys and dryad were used
#     # model_dir = "/home/gluetown/brain/data/embeddings/test/full/gys_dryad_point_net_not_group_kfold_epoch_50_num_layers1_hidden_dim_256_seed42_early_stopping_dropout/"
#     # with open(model_dir + "crit_idx.json", "r") as f:
#     #     critical_indices  = json.load(f)

#     # # get critical indices for best fold
#     # # best_idx = {k:v[0] for k, v in critical_indices["train"]["0"].items()}
#     # # for k, v in critical_indices["val"]["0"].items():
#     # #     if k in best_idx:
#     # #         best_idx[k].extend(v)
#     # #     else:
#     # #         best_idx[k] = v

#     # best_idx = {k: flatten_list(v) for k,v in critical_indices["train"]["0"].items()}
            
#     # best_proteins = {s:np.take(protein_labels1[s], idx).tolist() for s, idx in best_idx.items()}
#     # for key, value in best_proteins.items():
#     #     if not isinstance(value, list):
#     #         print(key)
#     #         best_proteins[key] = [value]
#     # #     # uniref90 = pd.read_csv("/home/gluetown/brain/data/uniref/uniref90_ids.txt", header=None)
#     # #     # uniref_dict ={}
#     # #     # for species in best_proteins.keys():
#     # #     #     uniref_dict[species] = [get_uniref90_id(uniprot_id) for uniprot_id in best_proteins[species]]


#     # #     # inter = {species:{} for species in best_proteins.keys()}
#     # #     # for species in best_proteins.keys():
#     # #     #     for species2 in best_proteins.keys():
#     # #     #         if species2 != species:
#     # #     #             inter[species][species2] = list(set(best_proteins[species]).intersection(set(best_proteins[species2])))
                

# #     # human: UP000005640_9606
# #     # mouse: UP000000589_10090
# #     for k,v in best_proteins.items():
# #         if type(v[0]) == list:
# #             best_proteins[k] = v[0]
# #     inter = flatten_list([set(best_proteins[s]).intersection(set(all_brain_genes)) for s in best_proteins.keys()])
# #     species_name_dict = pd.read_csv("/home/gluetown/brain/data/common_names.csv").set_index("Proteome_ID")["Common Name"].to_dict()

# # #     # Perform UMAP on individual species
# #     output_dir = "/home/gluetown/brain/data/embeddings/umap/brain_umap/" + model_dir.split("/")[-1] + "/"
# #     if not os.path.exists(output_dir):
# #         os.makedirs(output_dir)
# #     for species, species_id in zip(data, species_labels):
# #         try:
# #             onehot = []
# #             onehot_brain = []
# #             onehot = [1 if i in best_idx[species_id] else 0 for i in range(species.shape[0])]
# #             onehot_brain = [1 if i in all_brain_genes else 0 for i in protein_label_dict[species_id]]
# #             if any(onehot_brain):
# #                 print(species_id)
# #                 reducer = umap.UMAP(n_components=2, random_state=42)
# #                 embedding = reducer.fit_transform(species)
# #                 df = pd.DataFrame(embedding, columns=['UMAP1', 'UMAP2'])
# #                 # one hot critical indices
# #                 # only print umaps for species that have brain gene annotations
# #                 # df.to_csv(f"/home/gluetown/brain/data/embeddings/umap/{model_dir.split("/")[-1]}/{species_id}_umap.csv")
# #                 df['critical_indices'] = onehot     
# #                 df["brain"] = onehot_brain
# #                 df["color"] = df.apply(lambda row: 'orange' if row['critical_indices'] == 1 and row['brain'] == 1 else ('blue' if row['critical_indices'] == 1 else ('green' if row['brain'] == 1 else 'gray')), axis=1)
# #                 # orange = both, blue = critical idx, green = brain
# #                 legend_labels = {
# #                     'gray': 'Neither',
# #                     'blue': 'Critical Indices',
# #                     'green': 'Brain Genes',
# #                     'orange': 'Both'
# #                 }
# #                 handles = [plt.Line2D([0], [0], marker='o', color=color, markerfacecolor=color, markersize=2) for color in legend_labels.keys()]
# #                 labels = list(legend_labels.values())
# #                 plt.figure(figsize=(30, 30))
# #                 for color, label in legend_labels.items():
# #                     subset = df[df['color'] == color]
# #                     alpha = 0.5 if color == 'gray' else 1.0
# #                     plt.scatter(subset['UMAP1'], subset['UMAP2'], c=color, label=label, alpha=alpha, edgecolor='w', s=100)

# #                 # sns.scatterplot(data=df, x='UMAP1', y='UMAP2', hue='color', palette=['gray', 'blue', 'green', 'orange'])
# #                 plt.title(f'UMAP Projection Colored by Species | {species_id} | {species_name_dict[species_id.split("_")[0]]}')
# #                 plt.xlabel('UMAP 1')
# #                 plt.ylabel('UMAP 2')
# #                 plt.legend(handles, labels, title='Legend')
# #                 plt.show()
# #                 plt.savefig(f"{output_dir}/{species_id}.png")
# #                 print(f"Saved to {output_dir}/{species_id}.png")
# #         except: 
# #             print(f"Error with {species_id}")

        
# # Perform UMAP on order
#     # create dict of order: [species ...]
#     # loop through
#     # get onehot of crit idx for each species
# #     # vstack crit idx & vstack gene matrices

# #     # combine critical indices and map to combined UMAP
# #     combined_onehot = []
# #     flattened_list = [tensor[i,:] for tensor in data for i in range(tensor.shape[0])]
# #     # combined_df = pd.read_csv("/home/gluetown/brain/data/embeddings/umap.csv.gz", compression = "gzip")
# #     reducer = umap.UMAP(n_components=2, random_state=42)
# #     embedding = reducer.fit_transform(flattened_list)
# #     combined_df = pd.DataFrame(embedding, columns=['UMAP1', 'UMAP2'])
# #     combined_df.to_csv("/home/gluetown/brain/data/embeddings/umap.csv.gz", compression = "gzip")
# #     for species, species_id in zip(data, species_labels):
# #         try:
# #             onehot = [1 if i in best_idx[species_id] else 0 for i in range(species.shape[0])]
# #             combined_onehot.extend(onehot)
# #         except:
# #             pass
# #     combined_df['critical_indices'] = combined_onehot
# #     plt.figure(figsize=(20,10))
# #     # sns.scatterplot(data=df, x='UMAP1', y='UMAP2')
# #     sns.scatterplot(data=combined_df, x='UMAP1', y='UMAP2', hue='critical_indices', palette='tab10')
# #     plt.title(f'ALL UMAP Projection Colored')
# #     plt.xlabel('UMAP 1')
# #     plt.ylabel('UMAP 2')
# #     # plt.legend(title='Species')
# #     plt.show()
# #     plt.savefig(f"/home/gluetown/brain/data/embeddings/umap/{model_dir.split("/")[-1]}/combined_umap.png")
# #     print(f"Saved to /home/gluetown/brain/data/embeddings/umap/{model_dir.split("/")[-1]}/combined_umap.png")
    

# #### check out best model ####### # ##### FEATURE ATTRIBUTION ### 

# if __name__ == "__main__":
# dryad_file = "/home/gluetown/brain/data/embeddings/h5_files/dryad_reg_out_order.h5"
# pad_x1, y1, order_labels1, vocab1, mask1, species_labels1, protein_labels1  = load_output(dryad_file, protein_labels = True, order = True)
# x_list1, y_list1, y_list_normalized1, y_list_standardized1, order_list1, mask1, species_labels1 =  initialize_dataset(pad_x1, y1, order_labels1, vocab1, mask1, species_labels1)
# gys_file = "/home/gluetown/brain/data/embeddings/h5_files/gys_reg_out_order.h5"
# pad_x2, y2, order_labels2, vocab2, mask2, species_labels2, protein_labels2  = load_output(gys_file, protein_labels = True, order = True)   
# x_list2, y_list2, y_list_normalized2, y_list_standardized2, order_list2, mask2, species_labels2 =  initialize_dataset(pad_x2, y2, order_labels2, vocab2, mask2, species_labels2)
# x_list = x_list1 + x_list2
# y_list_normalized = y_list_normalized1 + y_list_normalized2
# order_list = order_list1 + order_list2
# species_labels = species_labels1 + species_labels2
# protein_labels1.update(protein_labels2) 

# model_file = "/home/gluetown/brain/data/embeddings/test/full/gys_dryad_point_net_not_group_kfold_epoch_50_num_layers1_hidden_dim_256_seed42_early_stopping_dropout/model/0_0_model.pth"
# input_dim = x_list[0].shape[0]
# output_dim = 1
# batch_size = 10
# learning_rate = 1e-4
# num_folds = 10
# max_epochs = 50
# use_early_stopping = True
# hidden_dim = 256
# use_dropout = True
# num_layers = 1
# point_net_model = PointNetRegHead(first_dim=input_dim, global_features = hidden_dim, k = output_dim, num_layers = num_layers, use_dropout=use_dropout)
# optimizer = optim.Adam(point_net_model.parameters(), lr=1e-4)
# point_net_model.load_state_dict(torch.load(model_file, weights_only=True, map_location=torch.device('cpu')))
# print("Model loaded successfully with matching parameters.")
# print("Model's state_dict:")
# for param_tensor in point_net_model.state_dict():
#     print(param_tensor, "\t", point_net_model.state_dict()[param_tensor].size())

# check_dataloader = create_dataloader(x_list, y_list_normalized, species_labels, 1, is_train=False)
# actual, pred, crit_idxs = get_predictions(point_net_model, check_dataloader)
# predicted = {s:{} for s in species_labels}
# for ind, s in enumerate(species_labels):
#     predicted[s]["pred"] = pred[ind][0]
#     predicted[s]["actual"] = actual[ind]
#     predicted[s]["mse"] = mean_squared_error(pred[ind], [actual[ind]])
# # predicted[s] = mean_squared_error(pred[ind], [actual[ind]])
# min_mse_species = min(predicted, key=lambda s: predicted[s]['mse'])
# common_names = pd.read_csv("/home/gluetown/brain/data/common_names.csv").set_index("Proteome_ID")["Common Name"].to_dict()
# best_species = sorted(predicted.items(), key=lambda item: item[1]['mse'])[:10]
# species_names = {i:common_names[i.split("_")[0]] for i in species_labels}
# # UP000005640,homo sapiens (human),homo sapiens
# # UP000000589,mus musculus (mouse),mus musculus
# # UP000546235,caloenas nicobarica (nicobar pigeon),caloenas nicobarica () {'pred': 0.44855237, 'actual': 0.44968945, 'mse': 1.292946e-06}

# fasta_dict = {}
# # humans
# with open("/home/gluetown/brain/data/uniprot_files/gys_dict/UP000005640_9606.fasta.json", "r") as f:
#     fasta_dict["UP000005640_9606.fasta"] = json.load(f)

# # mice
# with open("/home/gluetown/brain/data/uniprot_files/gys_dict/UP000000589_10090.fasta.json", "r") as f:
#     fasta_dict["UP000000589_10090.fasta"] = json.load(f)

# # best prediction: pigeon
# with open("/home/gluetown/brain/data/uniprot_files/dryad_dict/UP000546235_187106.fasta.json", "r") as f:
#     fasta_dict["UP000546235_187106.fasta"] = json.load(f)


# tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
# esm_model = AutoModelForMaskedLM.from_pretrained("facebook/esm2_t6_8M_UR50D")
# esm_model = EsmModel.from_pretrained("facebook/esm2_t6_8M_UR50D")

# input = fasta_dict["UP000005640_9606.fasta"]["Q96P65"]
# inputs = tokenizer(input, return_tensors="pt")
# outputs = esm_model(**inputs)



# torch.manual_seed(123)
# embed = torch.nn.Embedding(6, 16)
# embedded_sentence = embed(sentence_int).detach()

# input_ids = inputs["input_ids"].clone().detach().requires_grad_(True)
# inputs = inputs.float()
# inputs.requires_grad = True
# emb = esm_model(inputs.long())
# last_hidden_states = emb.last_hidden_state
# torch.autograd.grad(emb, inputs)

# stacked_model = StackedEsmPointnet(first_dim=input_dim, global_features = hidden_dim, k = output_dim,use_dropout=use_dropout)
# input_x_gradient = InputXGradient(stacked_model)
# input_x_gradient = InputXGradient(esm_model)

# out = stacked_model(inputs)
# attribution = input_x_gradient.attribute(inputs)
#  # no target b/c regression?


# torch.autograd.grad(out, inputs)




# #### FEATURE ATTRIBUTION WITH INPUT X GRAD
# from esm import FastaBatchedDataset, pretrained, MSATransformer
# import torch
# model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t6_8M_UR50D")
# import torch
# import esm
# import sys
# sys.path.append("/home/gluetown/brain/scripts/esm/esm/model/")
# import esm2 
# import regex as re
# model_name = "esm2_t6_8M_UR50D"
# model_data, regression_data = _download_model_and_regression_data(model_name)

# def upgrade_state_dict(state_dict):
#     """Removes prefixes 'model.encoder.sentence_encoder.' and 'model.encoder.'."""
#     prefixes = ["encoder.sentence_encoder.", "encoder."]
#     pattern = re.compile("^" + "|".join(prefixes))
#     state_dict = {pattern.sub("", name): param for name, param in state_dict.items()}
#     return state_dict

# cfg = model_data["cfg"]["model"]
# state_dict = model_data["model"]
# state_dict = upgrade_state_dict(state_dict)
# alphabet = esm.data.Alphabet.from_architecture("ESM-1b")
# model = esm2.ESM2(
#     num_layers=cfg.encoder_layers,
#     embed_dim=cfg.encoder_embed_dim,
#     attention_heads=cfg.encoder_attention_heads,
#     alphabet=alphabet,
#     token_dropout=False,
# )
# # model, alphabet, state_dict

# # requires_grad – Boolean indicating whether the Variable has been created by a subgraph containing any Variable, that requires it. Can be changed only on leaf Variables

# # 



# def forward_wrapper(batch_tokens):
#     embeddings = model.embed_tokens(batch_tokens)
#     embeddings = embeddings.detach()
#     embeddings.requires_grad = True
#     outputs = model(embeddings, batch_tokens, repr_layers=[6], return_contacts=True)
#     return outputs["representations"][6][0, :, :]

# def forward_wrapper(embeddings):
#     outputs = model(embeddings, batch_tokens, repr_layers=[6], return_contacts=True)
#     return outputs["representations"][6]

# # Load ESM-2 model
# # model, alphabet = esm.pretrained.esm2_t6_8M_UR50D() # load_model_and_alphabet_hub("esm2_t6_8M_UR50D")
# batch_converter = alphabet.get_batch_converter()
# model.eval()  # disables dropout for deterministic results

# # test dataset
# # Prepare data (first 2 sequences from ESMStructuralSplitDataset superfamily / 4)
# data = [
#     ("protein1", "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"),
#     ("protein2", "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE"),
#     ("protein2 with mask","KALTARQQEVFDLIRD<mask>ISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE"),
#     ("protein3",  "K A <mask> I S Q"),
# ]
# # reduced human gene dataset (critical indicies)
# fasta_file = "/home/gluetown/brain/data/feature_attribution/fasta_files/UP000005640_9606_crit_genes.fasta"
# data = []
# for record in SeqIO.parse(fasta_file, "fasta"):
#     sequence_id = record.id
#     sequence = str(record.seq)
#     data.append((sequence_id, sequence))
#     proteins[sequence_id] = sequence
# # take subset for testing
# # full dataset: all human genes
# # fasta_file = "/home/gluetown/brain/data/feature_attribution/fasta_files/UP000005640_9606.fasta"
# # data = []
# # for record in SeqIO.parse(fasta_file, "fasta"):
# #     sequence_id = record.id
# #     sequence = str(record.seq)
# #     data.append((sequence_id, sequence))


# # get attribution
# def batch_list(input_list, batch_size=10):
#     return [input_list[i:i + batch_size] for i in range(0, len(input_list), batch_size)]

# batched_data = batch_list(data)
# all_attribution = []
# for i in range(320):
#     feature_attribution = []
#     for batch in batched_data:
#         batch_labels, batch_strs, batch_tokens = batch_converter(batch)
#         batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
#         embeddings = model.embed_tokens(batch_tokens)
#         embeddings = embeddings.detach()
#         embeddings.requires_grad = True
#         # out = model(embeddings, batch_tokens, repr_layers=[6], return_contacts=True)
#         input_x_gradient = InputXGradient(forward_wrapper)
#         attribution = input_x_gradient.attribute(embeddings, target = (0,i))
#         aggregated_attributions = attribution.sum(dim=-1).squeeze().detach().cpu().numpy()
#         # aggregated_attributions = aggregated_attributions / np.max(aggregated_attributions)
#         feature_attribution += list(aggregated_attributions)
#     #### NOT SURE ABOUT TARGET????? #####
#     all_attribution.append(feature_attribution)

# for sample in data:
#     tokens = [i for i in sample[1]]  # Replace with real token strings
#     attribution_wo_padding = aggregated_attributions[0][0:len(sample[1])]
#     plt.figure(figsize=(12, 6))
#     plt.scatter(range(len(tokens)), attribution_wo_padding, color="blue", s=1, alpha=0.7)
#     plt.xticks(ticks=range(len(tokens)), labels=tokens)
#     plt.title(f"{sample[0]}Aggregated Attributions per Token")
#     plt.xlabel("Tokens")
#     plt.ylabel("Attribution Score")
#     plt.grid(axis="y", linestyle="--", alpha=0.5)
#     plt.show()
#     plt.savefig(f"/home/gluetown/brain/data/feature_attribution/inputxgrad/human_test_{sample[0]}.png")








# # embeddings.shape: torch.Size([4, 73, 320])
# # attribution.shape : torch.Size([4, 73, 320])

# """
# >>> out["representations"][6].shape
# torch.Size([4, 73, 320])
# # if it was a classification task, would just pick the class
# # technically regression but output is T x 320 (t = token length)

# """
# outputs = model(embeddings, batch_tokens,  repr_layers=[6], return_contacts=True)  # shape: [batch, seq_length, embedding_dim]
# token_scores = outputs["representations"][6].mean(dim=-1)  # Reduce embedding dimension
# batch_size = 4
# target = [0] * batch_size  # Focus on the first token for each sequence
# attributions = input_x_gradient.attribute(batch_tokens, target=target)



# """
# >>> attribution = input_x_gradient.attribute(batch_tokens)
# Traceback (most recent call last):
#   File "<stdin>", line 1, in <module>
#   File "/home/gluetown/miniconda3/envs/esm2/lib/python3.12/site-packages/captum/log/__init__.py", line 42, in wrapper
#     return func(*args, **kwargs)
#            ^^^^^^^^^^^^^^^^^^^^^
#   File "/home/gluetown/miniconda3/envs/esm2/lib/python3.12/site-packages/captum/attr/_core/input_x_gradient.py", line 117, in attribute
#     gradients = self.gradient_func(
#                 ^^^^^^^^^^^^^^^^^^^
#   File "/home/gluetown/miniconda3/envs/esm2/lib/python3.12/site-packages/captum/_utils/gradient.py", line 113, in compute_gradients
#     assert outputs[0].numel() == 1, (
# AssertionError: Target not provided when necessary, cannot take gradient with respect to multiple outputs.
# >>> attribution = input_x_gradient.attribute(batch_tokens[0])
# Traceback (most recent call last):
#   File "<stdin>", line 1, in <module>
#   File "/home/gluetown/miniconda3/envs/esm2/lib/python3.12/site-packages/captum/log/__init__.py", line 42, in wrapper
#     return func(*args, **kwargs)
#            ^^^^^^^^^^^^^^^^^^^^^
#   File "/home/gluetown/miniconda3/envs/esm2/lib/python3.12/site-packages/captum/attr/_core/input_x_gradient.py", line 117, in attribute
#     gradients = self.gradient_func(
#                 ^^^^^^^^^^^^^^^^^^^
#   File "/home/gluetown/miniconda3/envs/esm2/lib/python3.12/site-packages/captum/_utils/gradient.py", line 112, in compute_gradients
#     outputs = _run_forward(forward_fn, inputs, target_ind, additional_forward_args)
#               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/home/gluetown/miniconda3/envs/esm2/lib/python3.12/site-packages/captum/_utils/common.py", line 482, in _run_forward
#     output = forward_func(
#              ^^^^^^^^^^^^^
#   File "<stdin>", line 5, in forward_wrapper
#   File "/home/gluetown/miniconda3/envs/esm2/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1736, in _wrapped_call_impl
#     return self._call_impl(*args, **kwargs)
#            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/home/gluetown/miniconda3/envs/esm2/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1747, in _call_impl
#     return forward_call(*args, **kwargs)
#            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/home/gluetown/brain/scripts/esm/esm/model/esm2.py", line 80, in forward
#     assert tokens.ndim == 2
# AssertionError
# """





# """
# # w/o .detach()
# >>> embeddings = model.embed_tokens(batch_tokens)
# >>> embeddings.requires_grad = True
# Traceback (most recent call last):
#   File "<stdin>", line 1, in <module>
# RuntimeError: you can only change requires_grad flags of leaf variables.
# """
# ########### --> can only do requires_grad on leaf node
# ########## when any operation is performed on a tensor, it's no longer a leaf node




# # When runnign with token_dropout = True
# # should be regularization?? --> turned off when model.eval() is called
# # i dont understand why this is still runnihngt
# # model runs when toekn_dropout = False, but gradient is still not being calcualted
# """>>> out = model(embeddings, batch_tokens, repr_layers=[6], return_contacts=True)
# Traceback (most recent call last):
#   File "<stdin>", line 1, in <module>
#   File "/home/gluetown/miniconda3/envs/esm2/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1736, in _wrapped_call_impl
#     return self._call_impl(*args, **kwargs)
#            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/home/gluetown/miniconda3/envs/esm2/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1747, in _call_impl
#     return forward_call(*args, **kwargs)
#            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/home/gluetown/brain/scripts/esm/esm/model/esm2.py", line 86, in forward
#     x.masked_fill_((tokens == self.mask_idx).unsqueeze(-1), 0.0)
# RuntimeError: a leaf Variable that requires grad is being used in an in-place operation.
# >>> model = esm2.ESM2(
# ...     num_layers=cfg.encoder_layers,
# ...     embed_dim=cfg.encoder_embed_dim,
# ...     attention_heads=cfg.encoder_attention_heads,
# ...     alphabet=alphabet,
# ...     token_dropout=False,
# ... )
# >>> 
# """




# """
# >>> embeddings = model.embed_tokens(batch_tokens)
# >>> embeddings = embeddings.detach()
# >>> embeddings.requires_grad = True
# >>> out = model(embeddings, batch_tokens, repr_layers=[6], return_contacts=True)
# Traceback (most recent call last):
#   File "<stdin>", line 1, in <module>
#   File "/home/gluetown/miniconda3/envs/esm2/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1736, in _wrapped_call_impl
#     return self._call_impl(*args, **kwargs)
#            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/home/gluetown/miniconda3/envs/esm2/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1747, in _call_impl
#     return forward_call(*args, **kwargs)
#            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/home/gluetown/brain/scripts/esm/esm/model/esm2.py", line 86, in forward
#     x.masked_fill_((tokens == self.mask_idx).unsqueeze(-1), 0.0)
# RuntimeError: a leaf Variable that requires grad is being used in an in-place operation.
# """




# grad = torch.autograd.grad(
#     outputs=out["representations"][6],
#     inputs=embeddings,
#     grad_outputs=torch.ones(out["representations"][6].size()).to("cpu"), # or simply None if out is a scalar
#     retain_graph=False,
#     create_graph=False,
#     only_inputs=True)[0]

# """
# >>> grad = torch.autograd.grad(
# ...     outputs=out["representations"][6],
# ...     inputs=embeddings,
# ...     grad_outputs=torch.ones(out["representations"][6].size()).to("cpu"), # or simply None if out is a scalar
# ...     retain_graph=False,
# ...     create_graph=False,
# ...     only_inputs=True)[0]
# Traceback (most recent call last):
#   File "<stdin>", line 1, in <module>
#   File "/home/gluetown/miniconda3/envs/esm2/lib/python3.12/site-packages/torch/autograd/__init__.py", line 496, in grad
#     result = _engine_run_backward(
#              ^^^^^^^^^^^^^^^^^^^^^
#   File "/home/gluetown/miniconda3/envs/esm2/lib/python3.12/site-packages/torch/autograd/graph.py", line 825, in _engine_run_backward
#     return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
#            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# RuntimeError: One of the differentiated Tensors appears to not have been used in the graph. Set allow_unused=True if this is the desired behavior.
# >>> 
# """

# # out["representations"][6].backward(torch.ones_like(out["representations"][6]))
# # input_x_grad = batch_tokens.grad * batch_tokens


# """
# >>> out = model(batch_tokens, repr_layers=[6], return_contacts=True)
# Traceback (most recent call last):
#   File "<stdin>", line 1, in <module>
#   File "/home/gluetown/miniconda3/envs/esm2/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1736, in _wrapped_call_impl
#     return self._call_impl(*args, **kwargs)
#            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/home/gluetown/miniconda3/envs/esm2/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1747, in _call_impl
#     return forward_call(*args, **kwargs)
#            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/home/gluetown/brain/scripts/esm/esm/model/esm2.py", line 83, in forward
#     x.requires_grad = True
#     ^^^^^^^^^^^^^^^
# RuntimeError: you can only change requires_grad flags of leaf variables.
# out["logits"].grad
# <stdin>:1: UserWarning: The .grad attribute of a Tensor that is not a leaf Tensor is being accessed. Its .grad attribute won't be populated during autograd.backward(). If you indeed want the .grad field to be populated for a non-leaf Tensor, use .retain_grad() on the non-leaf Tensor. If you access the non-leaf Tensor by mistake, make sure you access the leaf Tensor instead. See github.com/pytorch/pytorch/pull/30531 for more informations. (Triggered internally at /opt/conda/conda-bld/pytorch_1728945370933/work/build/aten/src/ATen/core/TensorBody.h:489.)
# out["representations"][6].grad
# >>> """





# def forward_wrapper(inputs):
#     model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
#     inputs = inputs.float()
#     inputs.requires_grad = True
#     int_inputs = inputs.long() 
#     outputs = model(int_inputs, repr_layers=[6], return_contacts=True)
#     return outputs["representations"][6][:, 0, 0]


# input_x_gradient = InputXGradient(forward_wrapper)
# batch_tokens.requires_grad = True
# attribution = input_x_gradient.attribute(batch_tokens)
# input_x_gradient = InputXGradient(model)
# embeddings = model
# x.requires_grad = True
# attribution = input_x_gradient.attribute(x_list[0].unsqueeze(0))
# # Print the attribution
# print(attribution)





# ### Point Net Input x Grad

# pointNet = PointNetRegHead2(first_dim=input_dim, global_features = hidden_dim, k = output_dim, num_layers = num_layers, use_dropout=use_dropout)
# input_x_gradient = InputXGradient(pointNet)
# x_list[0].requires_grad = True
# attribution = input_x_gradient.attribute(x_list[0].unsqueeze(0))
# attribution


# # Extract per-residue representations (on CPU)
# with torch.no_grad():
#     results = model(batch_tokens, repr_layers=[6], return_contacts=True)

# token_representations = results["representations"][6]

# # Generate per-sequence representations via averaging
# # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
# sequence_representations = []
# for i, tokens_len in enumerate(batch_lens):
#     sequence_representations.append(token_representations[i, 1 : tokens_len - 1].mean(0))

# # Look at the unsupervised self-attention map contact predictions
# import matplotlib.pyplot as plt
# for (_, seq), tokens_len, attention_contacts in zip(data, batch_lens, results["contacts"]):
#     plt.matshow(attention_contacts[: tokens_len, : tokens_len])
#     plt.title(seq)
#     plt.show()
#     plt.savefig("/home/gluetown/brain/data/feature_attribution/test.png")



# # Install necessary libraries
# # !pip install torch torchvision transformers huggingface_hub captum
# # Import libraries
# import torch
# from transformers import AutoModelForSequenceClassification, AutoTokenizer
# from captum.attr import InputXGradient
# import matplotlib.pyplot as plt
# import numpy as np
# # Load ESM2 model and tokenizer
# model_name = "facebook/esm2_t6_8M_UR50D"
# model = AutoModelForSequenceClassification.from_pretrained(model_name)
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# for param in model.parameters():
#     param.requires_grad = True
# def preprocess_sequence(sequence):
#     inputs = tokenizer(sequence, return_tensors="pt")
#     return inputs
# sequence = "MENSDSADLIEDTAACRYSDHEKLRQRQVDLGMLQ"
# inputs = preprocess_sequence(sequence)
# # Perform a forward pass
# outputs = model(**inputs)
# print(f"Model output logits: {outputs.logits}")
# # Use Captum Input x Gradient
# # Define a wrapper function to compute attributions
# def forward_func(inputs_embeds):
#     # Pass embeddings through the model
#     output = model(inputs_embeds=inputs_embeds, attention_mask=inputs["attention_mask"])
#     return output.logits
# # Get input embeddings
# input_embeds = model.get_input_embeddings()(inputs["input_ids"])
# # Initialize InputXGradient object
# input_x_gradient = InputXGradient(forward_func)
# # Compute attributions
# attributions = input_x_gradient.attribute(input_embeds, target=0)  # target class index
# # Aggregate attributions across the embedding dimensions
# aggregated_attributions = attributions.sum(dim=-1).squeeze().detach().cpu().numpy()
# # Visualize the attributions for each amino acid in the sequence
# tokenized_sequence = tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze())
# plt.bar(range(len(tokenized_sequence)), aggregated_attributions, tick_label=tokenized_sequence)
# plt.xlabel(“Amino Acid”)
# plt.ylabel(“Attribution Score”)
# plt.title(“Input x Gradient Attribution Scores for Protein Sequence”)
# plt.show()


# sys.path.append("/home/gluetown/brain/scripts/esm/esm/model/")
# import 







# python interplm/train/train_plm_sae.py \
#     --plm_embd_dir /home/gluetown/brain/data/embeddings/final_embeddings/UP000005640_9606.fasta/ \
#     --save_dir models/walkthrough_model/




# with open("/home/gluetown/brain/data/uniref/all_genes/uniref.json", "r") as f:
#     uniref = json.load(f)
# Counter([len(i) for i in uniref.values()])
# Counter({1: 47192, 2: 5889, 3: 4441, 4: 3423, 5: 2518, 6: 1700, 7: 1138, 8: 800, 
# 9: 542, 10: 358, 11: 231, 12: 181, 13: 114, 15: 69, 14: 63, 16: 34, 17: 32, 19: 28, 
# 18: 27, 20: 15, 21: 10, 24: 9, 22: 8, 23: 6, 30: 5, 32: 4, 25: 4, 29: 3, 35: 3, 
# 27: 3, 37: 2, 26: 2, 48: 1, 43: 1, 50: 1, 136: 1, 46: 1, 86: 1, 28: 1, 34: 1, 
# 234: 1, 267: 1, 98: 1, 42: 1, 40: 1, 128: 1, 119: 1})

# # filtered uniref
# filtered_uniref = {k: v for k, v in uniref.items() if len(v) > 50}
# # get species
# metadata = pd.read_csv("/home/gluetown/brain/data/metadata_all_2.csv.gz", compression = "gzip")
# # test one cluster
# cluster1 = filtered_uniref[list(filtered_uniref.keys())[0]]
# # match to species
# for species, tmp in metadata.group_by("Species"):
# cluster1_species = {species:tmpfor species, tmp in metadata.group_by("Species")}




# dryad_file = "/home/gluetown/brain/data/embeddings/h5_files/dryad_reg_out_order.h5"
# pad_x1, y1, order_labels1, vocab1, mask1, species_labels1, protein_labels1  = load_output(dryad_file, protein_labels = True, order = True)
# x_list1, y_list1, y_list_normalized1, y_list_standardized1, order_list1, mask1, species_labels1 =  initialize_dataset(pad_x1, y1, order_labels1, vocab1, mask1, species_labels1)
# gys_file = "/home/gluetown/brain/data/embeddings/h5_files/gys_reg_out_order.h5"
# pad_x2, y2, order_labels2, vocab2, mask2, species_labels2, protein_labels2  = load_output(gys_file, protein_labels = True, order = True)   
# x_list2, y_list2, y_list_normalized2, y_list_standardized2, order_list2, mask2, species_labels2 =  initialize_dataset(pad_x2, y2, order_labels2, vocab2, mask2, species_labels2)
# x_list = x_list1 + x_list2
# y_list_normalized = y_list_normalized1 + y_list_normalized2
# order_list = order_list1 + order_list2
# species_labels = species_labels1 + species_labels2
# protein_labels1.update(protein_labels2) 

# # new uniref files
# for uniref_cluster, protein_ids in filtered_uniref.items():
#     protein_idx = {species: np.where(np.isin(v, protein_ids))[0] for species, v in protein_labels1.items()}
#     print(Counter([len(i) for i in protein_idx.values()]))
#     # for species_idx in protein_idx.keys():
#     #     new_x = [i[species_idx] for i in x_list]
#     #     new_y = [i[species_idx] for i in new_y]
#     #     new_species_labels = [i[species_idx] for i in species_labels]
#     #     new_order_list = [i[species_idx] for i in order_list]



# model_file = "/home/gluetown/brain/data/embeddings/test/full/gys_dryad_point_net_not_group_kfold_epoch_50_num_layers1_hidden_dim_256_seed42_early_stopping_dropout/model/0_0_model.pth"
# input_dim = x_list[0].shape[0]
# output_dim = 1
# batch_size = 10
# learning_rate = 1e-4
# num_folds = 10
# max_epochs = 50
# use_early_stopping = True
# hidden_dim = 256
# use_dropout = True
# num_layers = 1
# point_net_model = PointNetRegHead(first_dim=input_dim, global_features = hidden_dim, k = output_dim, num_layers = num_layers, use_dropout=use_dropout)
# optimizer = optim.Adam(point_net_model.parameters(), lr=1e-4)
# point_net_model.load_state_dict(torch.load(model_file, weights_only=True, map_location=torch.device('cpu')))
# print("Model loaded successfully with matching parameters.")
# print("Model's state_dict:")
# for param_tensor in point_net_model.state_dict():
#     print(param_tensor, "\t", point_net_model.state_dict()[param_tensor].size())




# # Process fasta files
# # datasets = ["gys", "dryad", "tsuboi"]
# # for dataset in datasets:
# #     uniprot_folder = f"/home/gluetown/brain/data/uniprot_files/{dataset}/"
# #     output_folder = f"/home/gluetown/brain/data/uniprot_files/{dataset}_dict/"
# #     if not os.path.exists(output_folder):
# #         os.mkdir(output_folder)

# #     for file in os.listdir(uniprot_folder):
# #         input_file = uniprot_folder + file
# #         output_file = f"{output_folder}/{file}.json"
# #         process_fasta(input_file, output_file)




# #### UNIREF Clusters ####
# # uniprot_ids = list(pd.read_csv("/home/gluetown/brain/data/weak_sup/all_genes.csv")["uniprot_id"].values)
# # map_uniprot_uniref(uniprot_ids, "/home/gluetown/brain/data/uniref/all_genes/")






# # # ############ MIL MODEL I FOUND ONLINE #######################################################################################


# # # ##########################################################################################################################################
 

# # # #############################################################################################################################
# # # ################### FUNCTIONS ###############################################################################################
# # # #############################################################################################################################

# # # class PointNetClassHead(nn.Module):
# # #     #Classification Head
# # #     def __init__(self, first_dim=40, second_dim = 64, conv1d_dims=[64], fc_blocks=[256], global_features=256, k=40):
# # #         super(PointNetClassHead, self).__init__()
# # #         # get the backbone (only need global features for classification)
# # #         self.backbone = PointNetBackbone(first_dim, second_dim, conv1d_dims=conv1d_dims, fc_blocks=fc_blocks, global_features = global_features)
# # #         # MLP for classification
# # #         self.linear = nn.Linear(global_features, 256)
# # #         self.out = nn.Linear(256, k)
# # #         # batchnorm for the first linear layers
# # #         # self.bn = nn.BatchNorm1d(256)
# # #         # The paper states that batch norm was only added to the layer 
# # #         # before the classification layer, but another version adds dropout  
# # #         # self.dropout = nn.Dropout(p=0.2)
# # #     def forward(self, x):
# # #         # get global features
# # #         x, crit_idxs, A_feat = self.backbone(x) 
# # #         print(x.shape)
# # #         x = F.relu(self.linear(x))
# # #         print(x.shape)
# # #         # x = self.bn(x)
# # #         # x = self.dropout(x)
# # #         embedding = x.clone()
# # #         x = self.out(x)
# # #         # return logits
# # #         return x


# # class BagModel(nn.Module):  
# #     def __init__(self, prepNN, afterNN, aggregation_func, model_type = 'reg', verbal=False):
# #         super().__init__()
# #         self.prepNN = prepNN
# #         self.aggregation_func = aggregation_func
# #         self.afterNN = afterNN
# #         self.verbal = verbal
# #         self.model_type = model_type
# #     def forward(self, input):  
# #         ids = input[1]
# #         input = input[0]
# #         # Modify shape of bagids if only 1d tensor
# #         if (len(ids.shape) == 1):
# #             ids.resize_(1, len(ids))
# #         inner_ids = ids[len(ids)-1]
# #         device = input.device
# #         NN_out = self.prepNN(input)
# #         unique, inverse, counts = torch.unique(inner_ids, sorted = True, return_inverse = True, return_counts = True)
# #         idx = torch.cat([(inverse == x).nonzero()[0] for x in range(len(unique))]).sort()[1]
# #         bags = unique[idx]
# #         counts = counts[idx]
# #         if self.verbal: print(f"NN_out: {NN_out.shape}, idx: {idx.shape}, bags: {bags}, counts: {counts}")
# #         output = torch.empty((len(bags), len(NN_out[0])), device = device)
# #         for i, bag in enumerate(bags):
# #             output[i] = self.aggregation_func(NN_out[inner_ids == bag], dim = 0) # shape = # bags, num_neurons
# #             if self.verbal: print(f"Aggregation: output: {output.shape}")
# #         output = self.afterNN(output)
# #         # if self.model_type == "class": output = np.round(output)
# #         if self.verbal: print(f"output: {output.shape}, ids: {ids.shape}")
# #         if (ids.shape[0] == 1):
# #             return output
# #         else:
# #             # I think this is for bag of bags model
# #             ids = ids[:len(ids)-1]
# #             mask = torch.empty(0, device = device).long()
# #             for i in range(len(counts)):
# #                 mask = torch.cat((mask, torch.sum(counts[:i], dtype = torch.int64).reshape(1)))
# #             if self.verbal: print(f"mask: {mask.shape}, ids: {ids.shape}")
# #         return (output, ids[:,mask])


# # def initialize_bags(pad_x, mask, y, order_labels):
# #     x_list = []
# #     for x in remove_mask(pad_x, mask):
# #         x_list.append(x)
# #     data = torch.cat(x_list, dim=0)
# #     label_dict = {i:j.item() for i, j in enumerate(y)}
# #     ids = []
# #     instance_labels = []
# #     for i, j in enumerate(x_list):
# #         for n in range(j.shape[0]):
# #             ids.append(i)
# #             instance_labels.append(label_dict[i])
# #     ids = torch.tensor(ids)
# #     bagids = torch.unique(ids)
# #     # labels = torch.stack([max(instance_labels[ids==i]) for i in bagids]).float()
# #     instance_labels = torch.tensor(instance_labels)
# #     labels = y
# #     print('INFO: Data shape \n  data: {}\n  ids: {}\n  labels: {}'.format(data.shape, ids.shape, labels.shape))
# #     dataset = mil.MilDataset(data, ids, labels)
# #     # train_indices, test_indices = train_test_split(np.arange(len(dataset)), test_size=0.2, stratify=order_labels)
# #     # train_indices, test_indices = train_test_split(np.arange(len(dataset)), test_size=0.2)
# #     # janky k fold split
# #     train_grp_indices = random.sample(list(set(order_labels)), int(len(set(order_labels)) - (int(len(set(order_labels)) / 5))))
# #     test_grp_indices = set(order_labels) - set(train_grp_indices)
# #     train_indices = []
# #     for i in train_grp_indices:
# #         train_indices += get_indices(i, order_labels)
# #     test_indices = []
# #     for i in test_grp_indices:
# #         test_indices +=  get_indices(i, order_labels)
# #     train, test = Subset(dataset, train_indices), Subset(dataset, test_indices)
# #     train_dl, test_dl = DataLoader(train, batch_size=batch_size, collate_fn=mil.collate, drop_last=True, shuffle=True), \
# #                         DataLoader(test, batch_size=batch_size, collate_fn=mil.collate, drop_last=True, shuffle=True)
# #     return dataset, train_dl, test_dl

# # def confusion_matrix(fold_actuals, fold_preds, fold_train_actuals, fold_train_preds, train_spearmans, val_spearmans, output_dir):
# #     num_folds = len(fold_actuals)
# #     for fold in range(num_folds):
# #         val_spearman = val_spearmans[fold]
# #         train_spearman = train_spearmans[fold]
# #         train_confusion_matrix = metrics.confusion_matrix(fold_train_actuals[fold], fold_train_preds[fold])
# #         val_confusion_matrix = metrics.confusion_matrix(fold_actuals[fold], fold_preds[fold])
# #         # g.set_axis_labels('Observed Brain Size Residuals', 'Predicted Brain Size Residuals')
# #         # Add Spearman correlation annotations
# #         # g.ax_joint.text(0.05, 0.95, f'Train Spearman: {train_spearman:.2f}', transform=g.ax_joint.transAxes, fontsize=12, verticalalignment='top', color='blue')
# #         # g.ax_joint.text(0.05, 0.90, f'Train mean: {statistics.mean(data[data["Type"] == "Training"]["Predicted"]):.2f}', transform=g.ax_joint.transAxes, fontsize=12, verticalalignment='top', color='blue')
# #         # g.ax_joint.text(0.05, 0.85, f'Train Std Dev: {statistics.stdev(data[data["Type"] == "Training"]["Predicted"]):.2f}', transform=g.ax_joint.transAxes, fontsize=12, verticalalignment='top', color='blue')
# #         # g.ax_joint.text(0.05, 0.80, f'Val Spearman: {val_spearman:.2f}', transform=g.ax_joint.transAxes, fontsize=12, verticalalignment='top', color='orange')
# #         fig, axes = plt.subplots(1, 2, figsize=(12, 6))
# #         cm_display_train = metrics.ConfusionMatrixDisplay(confusion_matrix=train_confusion_matrix, display_labels=[0, 1])
# #         cm_display_val = metrics.ConfusionMatrixDisplay(confusion_matrix=val_confusion_matrix, display_labels=[0, 1])
# #         cm_display_train.plot(ax=axes[0])
# #         axes[0].set_title('Training Confusion Matrix')        
# #         cm_display_val.plot(ax=axes[1])
# #         axes[1].set_title('Validation Confusion Matrix')        
# #         plt.tight_layout()
# #         plt.show()
# #         plt.savefig(output_dir + f"Fold{fold+1}-confusion_matrix.png")
# #         plt.figure(figsize=(8, 6))
# #         # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted 0', 'Predicted 1'], yticklabels=['Actual 0', 'Actual 1'])
# #         plt.xlabel('Predicted')
# #         plt.ylabel('Actual')
# #         plt.title('Confusion Matrix')
# #         plt.show()
# #         plt.savefig(output_dir + f"Fold{fold+1}-plot.png")

# # def training_loop(train_dl, test_dl, criterion, optimizer, model, num_folds, max_epochs, output_dir):
# #     fold_actuals = []
# #     fold_preds = []
# #     fold_train_actuals = []
# #     fold_train_preds = []
# #     fold_train_losses = []
# #     train_losses = []
# #     fold_val_losses = []
# #     val_losses = []
# #     train_spearmans = []
# #     val_spearmans = []
# #     for fold in range(num_folds):
# #         print(f"Fold: {fold}")
# #         train_actual = []
# #         train_pred = []
# #         val_actual = []
# #         val_pred = []
# #         for epoch in tqdm(range(max_epochs)):
# #             for batch_data, bagids, batch_labels in train_dl:
# #                 pred = model((batch_data, bagids)).squeeze()
# #                 loss = criterion(pred, batch_labels)
# #                 train_losses.append(loss.item())
# #                 optimizer.zero_grad()
# #                 loss.backward() 
# #                 optimizer.step()
# #                 train_pred.extend(pred.detach().numpy())
# #                 train_actual.extend(batch_labels.detach().numpy())
# #             fold_train_actuals.append(train_actual)
# #             fold_train_preds.append(train_pred)
# #             with torch.no_grad():
# #                 for batch_data, bagids, batch_labels in test_dl:
# #                     pred = model((batch_data, bagids)).squeeze()
# #                     loss = criterion(pred, batch_labels)
# #                     val_losses.append(loss.item())
# #                     val_pred.extend(pred.detach().numpy())
# #                     val_actual.extend(batch_labels.detach().numpy())
# #         fold_train_losses.append(train_losses)
# #         fold_val_losses.append(val_losses)
# #         fold_actuals.append(val_actual)
# #         fold_preds.append(val_pred)
# #         print(f"Train actual: {train_actual}")
# #         print(f"Train pred: {train_pred}")
# #         print(f"Val actual: {val_actual}")
# #         print(f"Val pred: {val_pred}")
# #         plot_losses(fold, train_losses, val_losses, output_dir)
# #         train_spearmans.append(compute_spearman_correlation(train_actual, train_pred))
# #         val_spearmans.append(compute_spearman_correlation(val_actual, val_pred))
# #         log_memory_usage()  # Log memory usage at the end of each fold
# #         # plot_scatter(fold_actuals, fold_preds, fold_train_actuals, fold_train_preds, train_spearmans, val_spearmans, output_dir)
# #     # confusion_matrix(fold_actuals, fold_preds, fold_train_actuals, fold_train_preds, train_spearmans, val_spearmans, output_dir)
# #     return model, fold_actuals, fold_preds, fold_train_actuals, fold_train_preds, fold_train_losses, fold_val_losses, train_spearmans, val_spearmans

# # # ######## INITIALIZATION ########### 
# # n_neurons = 15
# # lr = 1e-3
# # n_epochs = 100
# # batch_size = 4
# # output_dir = "/home/gluetown/brain/data/embeddings/test/mil/fnn_0/"
# # if not os.path.exists(output_dir):
# #     os.makedirs(output_dir)
# # # #### DATA ### 
# # # seed = 42 
# # # set_seed(seed)
# # # input_file = "../data/embeddings/test/go_terms/20.h5"
# # # pad_x, y, order_labels, vocab, mask = load_output(input_file, order=True)
# # # output_dir = "/home/gluetown/brain/data/embeddings/test/mil/point_net_20/"

# # pad_x = pad_x.float()
# # mask = mask.float()
# # y = y.float()

# # dataset, train_dl, test_dl = initialize_bags(pad_x, mask, y, order_labels)

# # # #### TRANSFORMER ######
# # # input_dim = 320
# # # num_heads = 4
# # # hidden_dim = 64
# # # num_layers = 1
# # # encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, dim_feedforward=hidden_dim, batch_first=True)
# # # transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
# # # prepNN = torch.nn.Sequential(transformer, 
# # #                              torch.nn.Linear(len(dataset.data[0]), n_neurons))

# # # ####### FNN ########
# # prepNN = torch.nn.Sequential(
# #   torch.nn.Linear(len(dataset.data[0]), n_neurons),
# #   torch.nn.ReLU()
# # )
# # afterNN = torch.nn.Sequential(
# #   torch.nn.Linear(n_neurons, 1)
# # )

# # # ######## POINT NET ########
# # # output_dir = "/home/gluetown/brain/data/embeddings/test/mil/point_net/"
# # # if not os.path.exists(output_dir):
# # #     os.makedirs(output_dir)
# # # global_features = 256
# # # first_dim = len(dataset.data[0])
# # # first_dim = 769
# # # second_dim = 64
# # # k = 15

# # # ### for point net , i guess it takes in 3d input
# # # # need to unconcatentate input and stack in 3d?
# # # # pad_x shape: torch.Size([96, 16, 320])
# # # backbone = PointNetClassHead(first_dim, second_dim, conv1d_dims=[64], fc_blocks=[256], global_features=global_features)
# # # linear = nn.Linear(global_features, 256)
# # # out = nn.Linear(256, k)
# # # prepNN = torch.nn.Sequential(
# # #   backbone, 
# # #   linear,
# # #   torch.nn.ReLU()
# # # )
# # # afterNN = torch.nn.Sequential(
# # #   torch.nn.Linear(k, 1)
# # # ) 

# # # ############################# TRAINING LOOP #############################
# # model = mil.BagModel(prepNN, afterNN, torch.mean)
# # criterion = nn.MSELoss()
# # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# # output_dir = f"/home/gluetown/brain/data/embeddings/test/mil_fnn_gys_max_epochs{max_epochs}/"
# # num_folds = 3
# # max_epochs = 10
# # model, fold_actuals, fold_preds, fold_train_actuals, fold_train_preds, fold_train_losses, fold_val_losses, train_spearmans, val_spearmans = training_loop(train_dl, test_dl, criterion, optimizer, model, num_folds, max_epochs, output_dir)
# # number_of_genes = mask.sum(dim=1)
# # print(compute_spearman_correlation(number_of_genes, y)) # 0.14300938754103606


# # # # baseline model
# # # # correlation between brain size and number of genes
# # # x_list = []
# # # for x in remove_mask(pad_x, mask):
# # #     x_list.append(x)
# # # number_of_genes = [len(i) for i in x_list]
# # # compute_spearman_correlation(number_of_genes, y) # -0.02607347555933114





# # #### Simpler classification task on gys dataset ##### 
# # n_neurons = 64
# # lr = 1e-3
# # n_epochs = 100
# # batch_size = 4

# # # load in data
# # input_file = "/group/gquongrp/collaborations/brain/embeddings.pt"
# # x_dict = torch.load(input_file)
# # max_len = max(tensor.shape[0] for tensor in x_dict.values())
# # pad_x = torch.zeros(len(x_dict.keys()), max_len, 320)
# # mask = torch.zeros(len(x_dict.keys()), max_len)
# # species_labels = []
# # for i, (species, tensor) in enumerate(x_dict.items()):
# #     species_labels.append(species)
# #     pad_x[i, :tensor.shape[0], :] = tensor
# #     mask[i, :tensor.shape[0]] = 1
# # # if len(mask.shape) != 2: 
# # #     mask = mask.unsqueeze(2)
# # labels_file = "/group/gquongrp/collaborations/brain/labels.pt"
# # y_dict = torch.load(labels_file)
# # y = [y_dict[k] for k in species_labels]
# # y = torch.tensor(y)
# # group_file = "/group/gquongrp/collaborations/brain/common_names.csv"
# # group_dict = pd.read_csv(group_file).set_index("Proteome_ID")["order"].to_dict()
# # order_labels = [group_dict[k.split("_")[0]] for k in species_labels]
# # vocab = {label: i for i, label in enumerate(set(order_labels))}
# # order_labels = [vocab[label] for label in order_labels]
# # # convert y to class (positive / negative)
# # pad_x = pad_x.float()
# # mask = mask.float()
# # # classifying if residual is positive / negative
# # y_class = torch.tensor([1 if i > 0 else 0 for i in y]).float() 
# # # classifying if residual is >1 std dev from mean
# # scaler = StandardScaler()
# # std_y = scaler.fit_transform(y.reshape(-1, 1))
# # y_class = torch.tensor([1 if i > 1 else 0 for i in std_y]).float() 
# # dataset, train_dl, test_dl = initialize_bags(pad_x, mask, y_class, order_labels)

# # prepNN = torch.nn.Sequential(
# #   torch.nn.Linear(len(dataset.data[0]), n_neurons),
# #   torch.nn.ReLU()
# # )
# # afterNN = torch.nn.Sequential(
# #     torch.nn.Linear(n_neurons, 1), 
# #     torch.nn.Sigmoid()
# # )
# # model = BagModel(prepNN, afterNN, torch.mean)
# # criterion = nn.CrossEntropyLoss()
# # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# # num_folds = 3
# # max_epochs = 30
# # output_dir = f"/home/gluetown/brain/data/embeddings/test/mil_fnn_gys_class_pos_neg_max_epochs{max_epochs}_hidden_dim{n_neurons}_class_std/"
# # if not os.path.exists(output_dir):
# #     os.makedirs(output_dir)
# # model, fold_actuals, fold_preds, fold_train_actuals, fold_train_preds, fold_train_losses, fold_val_losses, train_spearmans, val_spearmans = training_loop(train_dl, test_dl, criterion, optimizer, model, num_folds, max_epochs, output_dir)
# # # confusion_matrix(fold_actuals, fold_preds, fold_train_actuals, fold_train_preds, train_spearmans, val_spearmans, output_dir)

# # fold_preds = np.round(fold_preds)
# # fold_train_preds = np.round(fold_train_preds)





# # # # Create 4 instances divided to 2 bags in 3:1 ratio. First bag has positive label, second bag has negative label
# # # instances = torch.tensor([[1.0, 1.0, 1.0, 1.0],
# # # 			  [2.0, 2.0, 2.0, 2.0],
# # # 			  [3.0, 3.0, 3.0, 3.0],
# # # 			  [4.0, 4.0, 4.0, 4.0]])
# # # ids = torch.tensor([0, 0, 0, 1])
# # # labels = torch.tensor([1.0, 0.0])
# # # dataset = MilDataset(instances, ids, labels)
# # # dataloader = DataLoader(dataset=dataset, batch_size=batch_size, collate_fn=mil.collate)
# # # input_len = 4
# # # prepNN = torch.nn.Sequential(
# # #         torch.nn.Linear(input_len, 10),
# # #         torch.nn.ReLU(),
# # #     )
# # # afterNN = torch.nn.Sequential(
# # #         torch.nn.Linear(10, 1)
# # #     )
# # # model = BagModel(prepNN, afterNN, torch.mean)
# # # input = (instances, bagids)
# # # output = model(input)














# # # pool = torch.nn.AdaptiveAvgPool2d((1, 320))

# # # model = torch.nn.MultiheadAttention(embed_dim=320, num_heads=1)

# # # model = CLAM_SB(gate = True, size_arg = "small", dropout = 0., k_sample=2, n_classes=320, 
# # #                 instance_loss_fn=nn.MSELoss(), subtyping=False, embed_dim=320)
# # # for i in x_list:
# # #     logits, Y_prob, Y_hat, A_raw, results_dict = model(i)
# # #     Y_hat = Y_hat.squeeze(-1).squeeze(-1)
# # #     print(f"logits: {logits}, {logits.shape}\n Y_prob: {Y_prob}, {Y_prob.shape}\n Y_hat: {Y_hat}, {Y_hat.shape}\n A_raw: {A_raw}, {A_raw.shape}\n ")



# # # class Attn_Net_Gated(nn.Module):
# # #     def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 100):
# # #         super(Attn_Net_Gated, self).__init__()
# # #         self.attention_a = [
# # #             nn.Linear(L, D),
# # #             nn.Tanh()]        
# # #         self.attention_b = [nn.Linear(L, D),
# # #                             nn.Sigmoid()]
# # #         if dropout:
# # #             self.attention_a.append(nn.Dropout(0.25))
# # #             self.attention_b.append(nn.Dropout(0.25))
# # #         self.attention_a = nn.Sequential(*self.attention_a)
# # #         self.attention_b = nn.Sequential(*self.attention_b)        
# # #         self.attention_c = nn.Linear(D, n_classes)
# # #     def forward(self, x):
# # #         a = self.attention_a(x)
# # #         b = self.attention_b(x)
# # #         A = a.mul(b)
# # #         A = self.attention_c(A)  # N x n_classes
# # #         return A, x

# # # instance_loss_fn=nn.torch.nn.MSELoss()
# # # subtyping=False
# # # embed_dim=320
# # # dropout = 0.
# # # size_dict = {"small": [embed_dim, 512, 256], "big": [embed_dim, 512, 384]}
# # # size_arg = "small"
# # # size = size_dict[size_arg]
# # # fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout)]
# # # attention_net = Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_classes = 320)
# # # fc.append(attention_net)
# # # attention_net = nn.Sequential(*fc)
# # # A, x = attention_net(x_list[0])

# # # class MIL_Dataset(Dataset):
# # #     def __init__(self, x, y):
# # #         self.x = x 
# # #         self.y = y 
# # #     def __len__(self):
# # #         return len(self.y)    
# # #     def __getitem__(self, idx):
# # #         return self.x[idx], self.y[idx]


# # # def create_dataloader(data, labels, batch_size, is_train):
# # #     dataset = MIL_Dataset(data, labels)
# # #     if is_train:
# # #         return DataLoader(dataset, batch_size=batch_size, shuffle=True)
# # #     else:
# # #         return DataLoader(dataset, batch_size=batch_size, shuffle=False)

# # # """
# # # args:
# # #     gate: whether to use gated attention network
# # #     size_arg: config for network size
# # #     dropout: whether to use dropout
# # #     k_sample: number of positive/neg patches to sample for instance-level training
# # #     dropout: whether to use dropout (p = 0.25)
# # #     n_classes: number of classes 
# # #     instance_loss_fn: loss function to supervise instance-level training
# # #     subtyping: whether it's a subtyping problem
# # # """


# # # def train_and_evaluate(data, labels, group, mask, input_dim, hidden_dim, batch_size, learning_rate, num_folds, max_epochs, early_stopping, output_dir):
# # #     # initialization
# # #     model = CLAM_SB(gate = True, size_arg = "small", dropout = 0., k_sample=2, n_classes=320,
# # #                 instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False, embed_dim=320)
# # #     dataset = MIL_Dataset(data, labels)
# # #     dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
# # #     loss_fn = nn.MSELoss()  
# # #     optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# # #     # training loop
# # #     fold_actuals = []
# # #     fold_preds = []
# # #     fold_train_actuals = []
# # #     fold_train_preds = []
# # #     fold_train_losses = []
# # #     train_losses = []
# # #     fold_val_losses = []
# # #     val_losses = []
# # #     train_spearmans = []
# # #     val_spearmans = []
# # #     kf = GroupKFold(n_splits=num_folds)
# # #     for fold, (train_idx, val_idx) in enumerate(kf.split(data, labels, group)):
# # #         print(f'Fold {fold+1}/{num_folds}')
# # #         log_memory_usage()  # Log memory usage at the start of each fold
# # #         train_data, val_data = data[train_idx], data[val_idx]
# # #         train_labels, val_labels = labels[train_idx], labels[val_idx]
# # #         train_pad_mask, val_pad_mask = mask[train_idx], mask[val_idx]
# # #         train_dataloader = create_dataloader(train_data, train_labels, train_pad_mask, batch_size, is_train=1)
# # #         val_dataloader = create_dataloader(val_data, val_labels, val_pad_mask, batch_size, is_train=0)
# # #         train_actual = []
# # #         train_pred = []
# # #         val_actual = []
# # #         val_pred = []
# # #         for epoch in tqdm(range(max_epochs)):
# # #             for x, y in train_dataloader:
# # #                 logits, Y_prob, Y_hat, A_raw, results_dict = model(x)
# # #                 Y_hat = Y_hat.squeeze(-1).squeeze(-1)
# # #                 loss = loss_fn(Y_hat, y)
# # #                 train_losses.append(loss.item())
# # #                 optimizer.zero_grad()
# # #                 loss.backward() 
# # #                 optimizer.step()
# # #                 train_pred.extend(Y_hat.detach().numpy())
# # #                 train_actual.extend(y.detach().numpy())
# # #             fold_train_actuals.append(train_actual)
# # #             fold_train_preds.append(train_pred)
# # #             for x, y in val_dataloader:
# # #                 logits, Y_prob, Y_hat.squeeze(-1).squeeze(-1), A_raw, results_dict = model(x)
# # #                 loss = loss_fn(y, Y_hat)
# # #                 val_losses.append(loss.item())
# # #                 val_pred.extend(Y_hat.detach().numpy())
# # #                 val_actual.extend(y.detach().numpy())
# # #         fold_train_losses.append(train_losses)
# # #         fold_val_losses.append(val_losses)
# # #         fold_actuals.append(val_actual)
# # #         fold_preds.append(val_pred)
# # #         print(f"Train actual: {train_actual}")
# # #         print(f"Train pred: {train_pred}")
# # #         print(f"Val actual: {val_actual}")
# # #         print(f"Val pred: {val_pred}")
# # #         train_spearmans.append(compute_spearman_correlation(train_actual, train_pred))
# # #         val_spearmans.append(compute_spearman_correlation(val_actual, val_pred))
# # #         plot_losses(fold, train_losses, val_losses, output_dir)
# # #         log_memory_usage()  # Log memory usage at the end of each fold
# # #     return model, fold_actuals, fold_preds, fold_train_actuals, fold_train_preds, fold_train_losses, fold_val_losses, train_spearmans, val_spearmans
# ###### Checking dims of backbone ##### 
# # # class PointNetBackbone(nn.Module):
# #     def __init__(self, first_dim=40, second_dim = 64, 
# #                  conv1d_dims=[64], 
# #                  fc_blocks=[256], 
# #                  global_features=1024):
# #         super(PointNetBackbone, self).__init__()
# #         self.num_global_feats = global_features
# #         self.conv1d_dims = conv1d_dims + fc_blocks + [global_features]
# #         self.fc_blocks = [int(global_features/2)] + fc_blocks
# #         # Spatial Transformer Networks (T-nets)
# #         self.tnet1 = scPointNet.Tnet(dim=first_dim, conv1d_dims=self.conv1d_dims, fc_blocks=self.fc_blocks)
# #         self.tnet2 = scPointNet.Tnet(dim=second_dim, conv1d_dims=self.conv1d_dims[1:], fc_blocks=self.fc_blocks)
# #         # shared MLP
# #         self.shared_mlp1 = scPointNet.SharedMLP(first_dim, 64)
# #         self.shared_mlp2 = scPointNet.SharedMLP(second_dim, global_features)
# #         self.bn = nn.BatchNorm1d(self.num_global_feats)
# #     def forward(self, x, mask=None):
# #         # get batch size
# #         print(f"x: {x.shape}")
# #         batch_size = x.shape[0]
# #         num_points = x.shape[2]
# #         print(f"Batch Size: {batch_size}\n num_points:{num_points}")
# #         # pass through first Tnet to get transform matrix
# #         A_input = self.tnet1(x) # so that padded regions dont affect learned transformation matrix
# #         print(f"A_input: {A_input.shape}")
# #         # perform first transformation across each point in the batch
# #         x = torch.bmm(x.transpose(2, 1), A_input).transpose(2, 1)
# #         print(f"x: {x.shape} after first transformation")
# #         # pass through first shared MLP
# #         x = self.shared_mlp1(x, mask=mask) 
# #         print(f"x: {x.shape} after first shared MLP")
# #         # pass through second Tnet to get transform matrix
# #         A_feat = self.tnet2(x)
# #         # perform second transformation across each (64 dim) feature in the batch
# #         x = torch.bmm(x.transpose(2, 1), A_feat).transpose(2, 1)
# #         print(f"x: {x.shape} after second transformation")
# #         # store local point features for segmentation head
# #         local_features = x.clone()
# #         # pass through second shared MLP
# #         x = self.shared_mlp2(x)
# #         print(f"x: {x.shape} after second  shared MLP")
# #         x = self.bn(x)
# #         print(f"x: {x.shape} after batchnorm")
# #         # get global feature vector and critical indexes
# #         global_features, critical_indexes = F.max_pool1d(x, kernel_size=num_points, return_indices=True)#.view(batch_size, -1)
# #         global_features = global_features.view(batch_size, -1)
# #         critical_indexes = critical_indexes.view(batch_size, -1)
# #         return global_features, critical_indexes, A_feat
# ### 
# # >>> backbone(x_list[0].unsqueeze(0))
# # x: torch.Size([1, 320, 13441])
# # Batch Size: 1
# #  num_points:13441
# # A_input: torch.Size([1, 320, 320])
# # x: torch.Size([1, 320, 13441]) after first transformation
# # x: torch.Size([1, 64, 13441]) after first shared MLP
# # x: torch.Size([1, 64, 13441]) after second transformation
# # x: torch.Size([1, 1024, 13441]) after second  shared MLP
# # x: torch.Size([1, 1024, 13441]) after batchnorm
# # (tensor([[ 7.9055, 10.0412, 12.4669,  ..., 11.0146,  7.2155,  8.6828]],
# #        grad_fn=<ViewBackward0>), tensor([[10326,  7623,  3465,  ...,  2454,  1531, 10481]]), tensor([[[ 0.3329, -0.4541, -0.3970,  ...,  0.1433,  0.5817, -0.6483],
# #          [-0.1896,  1.4743, -0.0594,  ...,  0.5242,  0.0286, -0.2734],
# #          [ 0.1497,  0.2458,  0.7439,  ...,  0.5450,  0.6818, -0.0210],
# #          ...,
# #          [ 0.6635,  0.1497,  0.2633,  ...,  1.8011,  0.1449,  0.7297],
# #          [ 0.1083,  0.1394, -0.2182,  ..., -0.1729,  0.3158, -0.0216],
# #          [-0.1507, -0.0287,  0.3992,  ..., -0.9897,  1.0568,  1.0905]]],
# #        grad_fn=<AddBackward0>))





