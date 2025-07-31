import pickle as pkl
import re
import urllib
import warnings
from argparse import Namespace
from pathlib import Path
from pytorch_lightning import seed_everything
import torch
import esm
from esm.model.esm2 import ESM2
from torch.utils.data import Dataset, DataLoader
from captum.attr import InputXGradient
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import AutoTokenizer, AutoModelForMaskedLM, EsmModel
import psutil
import matplotlib.pyplot as plt
import seaborn as sns
import pickle as pkl
sys.path.append("/home/gluetown/brain/scripts/scPointNet_dev/src/")
import scPointNet 
from Bio import SeqIO
from esm import FastaBatchedDataset, pretrained, MSATransformer
import torch
model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t6_8M_UR50D")
import torch
import esm
import random
import sys
sys.path.append("/home/gluetown/brain/scripts/esm/esm/model/")
import esm2 
import regex as re
from torch.cuda.amp import autocast, GradScaler
import h5py
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from collections import Counter

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

def get_predictions(model, dataloader):
    model.eval()
    all_actuals = []
    all_predictions = []
    with torch.no_grad():
        for batch in dataloader:
            process = psutil.Process(os.getpid())
            mem_before = process.memory_info().rss / (1024 ** 2)
            inputs, targets, species = batch
            outputs = model(inputs)
            all_actuals.extend(targets.cpu().numpy())
            all_predictions.extend(outputs.cpu().numpy())
            mem_after = process.memory_info().rss / (1024 ** 2)
            mem_used = mem_after - mem_before
            print(f"Memory used to get predictions: {mem_used:.2f} MB")
        all_actuals = np.array(all_actuals)
        all_predictions = np.array(all_predictions)        
        return all_actuals, all_predictions

def create_dataloader(data, labels, species, batch_size, is_train):
    dataset = ListDataset(data, labels, species)
    if is_train: 
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    else:
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)

class ListDataset(Dataset):
    def __init__(self, x, y, species):
        self.x = x
        self.y = y
        self.species = species
    def __len__(self):
        return(len(self.x))
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.species[idx]
        
def log_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f"Memory Usage: {mem_info.rss / (1024 ** 2):.2f} MB")

def batch_list(input_list, batch_size=10):
    return [input_list[i:i + batch_size] for i in range(0, len(input_list), batch_size)]

def _has_regression_weights(model_name):
    """Return whether we expect / require regression weights;
    Right now that is all models except ESM-1v, ESM-IF, and partially trained ESM2 models"""
    return not ("esm1v" in model_name or "esm_if" in model_name or "270K" in model_name or "500K" in model_name)

def load_model_and_alphabet(model_name):
    if model_name.endswith(".pt"):  # treat as filepath
        return load_model_and_alphabet_local(model_name)
    else:
        return load_model_and_alphabet_hub(model_name)

def load_hub_workaround(url):
    try:
        data = torch.hub.load_state_dict_from_url(url, progress=False, map_location="cpu")
    except RuntimeError:
        # Pytorch version issue - see https://github.com/pytorch/pytorch/issues/43106
        fn = Path(url).name
        data = torch.load(
            f"{torch.hub.get_dir()}/checkpoints/{fn}",
            map_location="cpu",
        )
    except urllib.error.HTTPError as e:
        raise Exception(f"Could not load {url}, check if you specified a correct model name?")
    return data

def load_regression_hub(model_name):
    url = f"https://dl.fbaipublicfiles.com/fair-esm/regression/{model_name}-contact-regression.pt"
    regression_data = load_hub_workaround(url)
    return regression_data

def _download_model_and_regression_data(model_name):
    url = f"https://dl.fbaipublicfiles.com/fair-esm/models/{model_name}.pt"
    model_data = load_hub_workaround(url)
    if _has_regression_weights(model_name):
        regression_data = load_regression_hub(model_name)
    else:
        regression_data = None
    return model_data, regression_data

def load_model_and_alphabet_hub(model_name):
    model_data, regression_data = _download_model_and_regression_data(model_name)
    return load_model_and_alphabet_core(model_name, model_data, regression_data)

def load_model_and_alphabet_local(model_location):
    """Load from local path. The regression weights need to be co-located"""
    model_location = Path(model_location)
    model_data = torch.load(str(model_location), map_location="cpu")
    model_name = model_location.stem
    if _has_regression_weights(model_name):
        regression_location = str(model_location.with_suffix("")) + "-contact-regression.pt"
        regression_data = torch.load(regression_location, map_location="cpu")
    else:
        regression_data = None
    return load_model_and_alphabet_core(model_name, model_data, regression_data)

def has_emb_layer_norm_before(model_state):
    """Determine whether layer norm needs to be applied before the encoder"""
    return any(k.startswith("emb_layer_norm_before") for k, param in model_state.items())

def _load_model_and_alphabet_core_v1(model_data):
    import esm  # since esm.inverse_folding is imported below, you actually have to re-import esm here

    alphabet = esm.Alphabet.from_architecture(model_data["args"].arch)

    if model_data["args"].arch == "roberta_large":
        # upgrade state dict
        pra = lambda s: "".join(s.split("encoder_")[1:] if "encoder" in s else s)
        prs1 = lambda s: "".join(s.split("encoder.")[1:] if "encoder" in s else s)
        prs2 = lambda s: "".join(
            s.split("sentence_encoder.")[1:] if "sentence_encoder" in s else s
        )
        model_args = {pra(arg[0]): arg[1] for arg in vars(model_data["args"]).items()}
        model_state = {prs1(prs2(arg[0])): arg[1] for arg in model_data["model"].items()}
        model_state["embed_tokens.weight"][alphabet.mask_idx].zero_()  # For token drop
        model_args["emb_layer_norm_before"] = has_emb_layer_norm_before(model_state)
        model_type = esm.ProteinBertModel

    elif model_data["args"].arch == "protein_bert_base":

        # upgrade state dict
        pra = lambda s: "".join(s.split("decoder_")[1:] if "decoder" in s else s)
        prs = lambda s: "".join(s.split("decoder.")[1:] if "decoder" in s else s)
        model_args = {pra(arg[0]): arg[1] for arg in vars(model_data["args"]).items()}
        model_state = {prs(arg[0]): arg[1] for arg in model_data["model"].items()}
        model_type = esm.ProteinBertModel
    elif model_data["args"].arch == "msa_transformer":

        # upgrade state dict
        pra = lambda s: "".join(s.split("encoder_")[1:] if "encoder" in s else s)
        prs1 = lambda s: "".join(s.split("encoder.")[1:] if "encoder" in s else s)
        prs2 = lambda s: "".join(
            s.split("sentence_encoder.")[1:] if "sentence_encoder" in s else s
        )
        prs3 = lambda s: s.replace("row", "column") if "row" in s else s.replace("column", "row")
        model_args = {pra(arg[0]): arg[1] for arg in vars(model_data["args"]).items()}
        model_state = {prs1(prs2(prs3(arg[0]))): arg[1] for arg in model_data["model"].items()}
        if model_args.get("embed_positions_msa", False):
            emb_dim = model_state["msa_position_embedding"].size(-1)
            model_args["embed_positions_msa_dim"] = emb_dim  # initial release, bug: emb_dim==1

        model_type = esm.MSATransformer

    elif "invariant_gvp" in model_data["args"].arch:
        import esm.inverse_folding

        model_type = esm.inverse_folding.gvp_transformer.GVPTransformerModel
        model_args = vars(model_data["args"])  # convert Namespace -> dict

        def update_name(s):
            # Map the module names in checkpoints trained with internal code to
            # the updated module names in open source code
            s = s.replace("W_v", "embed_graph.embed_node")
            s = s.replace("W_e", "embed_graph.embed_edge")
            s = s.replace("embed_scores.0", "embed_confidence")
            s = s.replace("embed_score.", "embed_graph.embed_confidence.")
            s = s.replace("seq_logits_projection.", "")
            s = s.replace("embed_ingraham_features", "embed_dihedrals")
            s = s.replace("embed_gvp_in_local_frame.0", "embed_gvp_output")
            s = s.replace("embed_features_in_local_frame.0", "embed_gvp_input_features")
            return s

        model_state = {
            update_name(sname): svalue
            for sname, svalue in model_data["model"].items()
            if "version" not in sname
        }

    else:
        raise ValueError("Unknown architecture selected")

    model = model_type(
        Namespace(**model_args),
        alphabet,
    )

    return model, alphabet, model_state

def _load_model_and_alphabet_core_v2(model_data):
    def upgrade_state_dict(state_dict):
        """Removes prefixes 'model.encoder.sentence_encoder.' and 'model.encoder.'."""
        prefixes = ["encoder.sentence_encoder.", "encoder."]
        pattern = re.compile("^" + "|".join(prefixes))
        state_dict = {pattern.sub("", name): param for name, param in state_dict.items()}
        return state_dict

    cfg = model_data["cfg"]["model"]
    state_dict = model_data["model"]
    state_dict = upgrade_state_dict(state_dict)
    alphabet = esm.data.Alphabet.from_architecture("ESM-1b")
    model = ESM2(
        num_layers=cfg.encoder_layers,
        embed_dim=cfg.encoder_embed_dim,
        attention_heads=cfg.encoder_attention_heads,
        alphabet=alphabet,
        token_dropout=cfg.token_dropout,
    )
    return model, alphabet, state_dict

def load_model_and_alphabet_core(model_name, model_data, regression_data=None):
    if regression_data is not None:
        model_data["model"].update(regression_data["model"])

    if model_name.startswith("esm2"):
        model, alphabet, model_state = _load_model_and_alphabet_core_v2(model_data)
    else:
        model, alphabet, model_state = _load_model_and_alphabet_core_v1(model_data)

    expected_keys = set(model.state_dict().keys())
    found_keys = set(model_state.keys())

    if regression_data is None:
        expected_missing = {"contact_head.regression.weight", "contact_head.regression.bias"}
        error_msgs = []
        missing = (expected_keys - found_keys) - expected_missing
        if missing:
            error_msgs.append(f"Missing key(s) in state_dict: {missing}.")
        unexpected = found_keys - expected_keys
        if unexpected:
            error_msgs.append(f"Unexpected key(s) in state_dict: {unexpected}.")

        if error_msgs:
            raise RuntimeError(
                "Error(s) in loading state_dict for {}:\n\t{}".format(
                    model.__class__.__name__, "\n\t".join(error_msgs)
                )
            )
        if expected_missing - found_keys:
            warnings.warn(
                "Regression weights not found, predicting contacts will not produce correct results."
            )

    model.load_state_dict(model_state, strict=regression_data is not None)
    return model, alphabet

def upgrade_state_dict(state_dict):
    """Removes prefixes 'model.encoder.sentence_encoder.' and 'model.encoder.'."""
    prefixes = ["encoder.sentence_encoder.", "encoder."]
    pattern = re.compile("^" + "|".join(prefixes))
    state_dict = {pattern.sub("", name): param for name, param in state_dict.items()}
    return state_dict

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
    
class StackedEsmPointnet(nn.Module):
    def __init__(self, model_data, model_file,  first_dim=320, second_dim = 64, conv1d_dims=[64], fc_blocks=[256], global_features=256, k=1, use_dropout=True, use_layer_norm=False):
        super(StackedEsmPointnet, self).__init__()
        self.pointNet = PointNetRegHead2(first_dim, second_dim, conv1d_dims, fc_blocks, global_features, k, use_dropout, use_layer_norm)
        self.pointNet.load_state_dict(torch.load(model_file, weights_only=True, map_location=torch.device('cpu')))
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
        cfg = model_data["cfg"]["model"]
        state_dict = model_data["model"]
        state_dict = upgrade_state_dict(state_dict)
        alphabet = esm.data.Alphabet.from_architecture("ESM-1b")
        self.esm_model = esm2.ESM2(
            num_layers=cfg.encoder_layers,
            embed_dim=cfg.encoder_embed_dim,
            attention_heads=cfg.encoder_attention_heads,
            alphabet=alphabet,
            token_dropout=False)
        self.batch_converter = alphabet.get_batch_converter()
    def forward(self, initial_embeddings, batch_tokens):
        # tokenized = self.tokenizer(x,return_tensors="pt")
        # better to tokenize before passing in, b/c captum only takes in tensors
        result = self.esm_model(initial_embeddings, batch_tokens, repr_layers=[6], return_contacts=True)
        mean_result = result["representations"][6].mean(dim=1)
        embedding = mean_result.unsqueeze(0).permute(0,2,1)
        out = self.pointNet(embedding)
        return out
    
def get_predictions(model, dataloader):
    model.eval()
    all_actuals = []
    all_predictions = []
    with torch.no_grad():
        for batch in dataloader:
            process = psutil.Process(os.getpid())
            mem_before = process.memory_info().rss / (1024 ** 2)
            inputs, targets, species = batch
            outputs = model(inputs)
            all_actuals.extend(targets.cpu().numpy())
            all_predictions.extend(outputs.cpu().numpy())
            mem_after = process.memory_info().rss / (1024 ** 2)
            mem_used = mem_after - mem_before
            print(f"Memory used to get predictions: {mem_used:.2f} MB")
        all_actuals = np.array(all_actuals)
        all_predictions = np.array(all_predictions)
        return all_actuals, all_predictions
    
def remove_mask(x, mask):
    # returns iterable of x (ragged)
    for species in range(mask.shape[0]): 
        bool_array = [True if i == 1 else False for i in mask[species]]
        yield x[species][bool_array]

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
                df = pd.DataFrame([entry.split(":") for entry in protein_labels], columns=["species", "uniprot_id"])
                protein_labels = df.groupby('species')['uniprot_id'].apply(list).to_dict()
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

def initialize_dataset(pad_x, y, order_labels, vocab, mask, species_labels):
    x_list = []
    for x in remove_mask(pad_x, mask):
        x_list.append(x.permute(1,0))
    trouble_makers = ["UP000189704_1868482.fasta", "UP000009136_9913.fasta", "UP000694520_30521.fasta"]
    troublemaker_indices = list(np.array([get_indices(i, species_labels) for i in trouble_makers]).flatten())
    y_list = list(y)    
    all_orders = order_labels
    order_list = [i if Counter(all_orders)[i] > 5 else "Other" for i in all_orders]    
    x_list = [x_list[i] for i in range(len(x_list)) if i not in troublemaker_indices]
    y_list_normalized = min_max_normalize(y_list)
    y_list_standardized = z_score_normalize(y_list)
    return x_list, y_list, y_list_normalized, y_list_standardized, order_list, mask, species_labels


def match_dict_indices(dict1, list2):
    # assuming keys are species and the values (lists) are what you're trying to match
    # gets inidices fo all occurences of list2 in dict1
    trouble_makers = ["UP000189704_1868482.fasta", "UP000009136_9913.fasta", "UP000694520_30521.fasta"]
    dict_indices = {k:[] for k in dict1.keys() if k not in trouble_makers}
    for k1, v1 in dict1.items():
        list2_idx = np.where(np.isin(list2, v1))[0]
        dict_indices[k1] = [v1.index(list2[i]) for i in list2_idx]
    return dict_indices

def subset_data(dict1, x_list, y_list, order_list, protein_labels, species_labels):
    dict1 = {k:v for k,v in dict1.items() if len(v) > 1}
    species = list(dict1.keys())
    species_idx = np.where(np.isin(species_labels, species))[0]
    new_y_list = np.take(np.array(y_list), species_idx).tolist()
    new_order_list = np.take(np.array(order_list), species_idx).tolist()
    new_species_labels = np.take(np.array(species_labels), species_idx).tolist()
    new_protein_labels = {}
    for k,v in dict1.items():
        new_protein_labels[k] = np.take(np.array(protein_labels[k]), v).tolist()
    protein_labels_idx = {k:np.where(np.isin(protein_labels[k], new_protein_labels[k]))[0] for k,v in new_protein_labels.items()}
    new_x_list = [x_list[i][:, protein_labels_idx[spe]] for i, spe in zip(species_idx, new_species_labels)]
    return new_x_list, new_y_list, new_order_list, new_species_labels, new_protein_labels

def get_indices(element, lst):
    return [i for i in range(len(lst)) if lst[i] == element]

def flatten_list(lis):
    return [item for sublist in lis for item in sublist]

def intialize_attribution_data(fasta_file, proteins):
    data = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        sequence_id = record.id
        sequence = str(record.seq)
        if sequence_id.split("|")[1] in proteins:
            data.append((sequence_id.split("|")[1], sequence))
            
    try:
        protein_names = [i[0].split("|")[1] for i in data]
    except:
        protein_names = [i[0] for i in data]
    return data, protein_names


# ### DEFINE MODEL ### 
# log_memory_usage()
# model_name = "esm2_t6_8M_UR50D"
# model_data, regression_data = _download_model_and_regression_data(model_name)
# model_file = "/home/gluetown/brain/data/embeddings/test/full/gys_dryad_point_net_not_group_kfold_epoch_50_num_layers1_hidden_dim_256_seed42_early_stopping_dropout/model/0_0_model.pth"
# input_dim = 320
# output_dim = 1
# batch_size = 10
# learning_rate = 1e-4
# num_folds = 10
# max_epochs = 50
# use_early_stopping = True
# hidden_dim = 256
# use_dropout = True
# num_layers = 1
# point_net_model = PointNetRegHead2(first_dim=input_dim, global_features = hidden_dim, k = output_dim, num_layers = num_layers, use_dropout=use_dropout)
# point_net_model.load_state_dict(torch.load(model_file, weights_only=True, map_location=torch.device('cpu')))
# print("Model loaded successfully with matching parameters.")
# print("Model's state_dict:")
# for param_tensor in point_net_model.state_dict():
#     print(param_tensor, "\t", point_net_model.state_dict()[param_tensor].size())

# cfg = model_data["cfg"]["model"]
# state_dict = model_data["model"]
# state_dict = upgrade_state_dict(state_dict)
# alphabet = esm.data.Alphabet.from_architecture("ESM-1b")
# esm_model = esm2.ESM2(
#     num_layers=cfg.encoder_layers,
#     embed_dim=cfg.encoder_embed_dim,
#     attention_heads=cfg.encoder_attention_heads,
#     alphabet=alphabet,
#     token_dropout=False,
# )
# stacked_model = StackedEsmPointnet(model_data, model_file, first_dim=input_dim, global_features = hidden_dim, k = output_dim,use_dropout=use_dropout)
# log_memory_usage()

# def forward_wrapper(batch_tokens):
#     embeddings = model.embed_tokens(batch_tokens)
#     embeddings = embeddings.detach()
#     embeddings.requires_grad = True
#     outputs = model(embeddings, batch_tokens, repr_layers=[6], return_contacts=True)
#     return outputs["representations"][6][0, :, :]

# def forward_wrapper(embeddings):
#     outputs = model(embeddings, batch_tokens, repr_layers=[6], return_contacts=True)
#     return outputs["representations"][6]

# def forward_wrapper(embeddings):
#     outputs = stacked_model(embeddings, batch_tokens)
#     return outputs


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
# # take subset for testing
# # full dataset: all human genes
# # fasta_file = "/home/gluetown/brain/data/feature_attribution/fasta_files/UP000005640_9606.fasta"
# # data = []
# # for record in SeqIO.parse(fasta_file, "fasta"):
# #     sequence_id = record.id
# #     sequence = str(record.seq)
# #     data.append((sequence_id, sequence))

# try:
#     protein_names = [i[0].split("|")[1] for i in data]
# except:
#     protein_names = [i[0] for i in data]


# # subset
# # data = data[0:10]
# batch_labels, batch_strs, batch_tokens = batch_converter(data)
# batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
# embeddings = model.embed_tokens(batch_tokens)
# log_memory_usage()
# input_x_gradient = InputXGradient(forward_wrapper)
# attribution = input_x_gradient.attribute(embeddings)
# # out = stacked_model(embeddings, batch_tokens)

# filehandler = open(b"/home/gluetown/brain/data/feature_attribution/inputxgrad/stacked_attribution.pkl","wb")
# pkl.dump(attribution,filehandler)



# ### BATCH PROCESSING ###
# if torch.cuda.is_available():
#     model = model.cuda()
#     print("Transferred model to GPU")

# dataset = FastaBatchedDataset.from_file(fasta_file)
# batches = dataset.get_batch_indices(4096, extra_toks_per_seq=1)
# data_loader = torch.utils.data.DataLoader(
#     dataset, collate_fn=alphabet.get_batch_converter(1022), batch_sampler=batches
# )
# print(f"Read {fasta_file} with {len(dataset)} sequences")
    
# output_dir.mkdir(parents=True, exist_ok=True)

# assert all(-(model.num_layers + 1) <= i <= model.num_layers for i in [0, 5, 6])
# repr_layers = [(i + model.num_layers + 1) % (model.num_layers + 1) for i in [0, 5, 6]]

# with torch.no_grad():
#     for batch_idx, (labels, strs, toks) in enumerate(data_loader):
#         print(
#             f"Processing {batch_idx + 1} of {len(batches)} batches ({toks.size(0)} sequences)"
#         )
#         if torch.cuda.is_available():
#             toks = toks.to(device="cuda", non_blocking=True)

#         out = model(toks, repr_layers=repr_layers, return_contacts=False)

#         logits = out["logits"].to(device="cpu")
#         representations = {
#             layer: t.to(device="cpu") for layer, t in out["representations"].items()
#         }

#         for i, label in enumerate(labels):
#             output_file = output_dir / f"{label}.pt"
#             print(output_file)
#             output_file.parent.mkdir(parents=True, exist_ok=True)
#             result = {"label": label}
#             truncate_len = min(1022, len(strs[i]))
#             # Call clone on tensors to ensure tensors are not views into a larger representation
#             # See https://github.com/pytorch/pytorch/issues/1995

#             result["mean_representations"] = {
#                 layer: t[i, 1 : truncate_len + 1].mean(0).clone()
#                 for layer, t in representations.items()
#             }

#             torch.save(
#                 result,
#                 output_file,
#             )
#     print(f"Finished writing embeddings to {output_dir}")






# # PLOTS

# ### pool across sequence (to see feature attribution)
# pooled_seq = attribution.mean(dim=1).detach().numpy()
# plt.figure(figsize=(30, 15))
# cax = plt.imshow(pooled_seq, aspect="auto", cmap="rocket_r", interpolation='nearest')
# plt.colorbar(cax, label="Value")
# plt.xlabel("Features", fontsize=24)         
# plt.ylabel("Genes", fontsize=24)      
# plt.title(f"Best predicted species attribution over features", fontsize=30)
# plt.grid(visible=False)
# plt.xticks(range(pooled_seq.shape[1]), fontsize=30)
# plt.yticks(range(pooled_seq.shape[0]), labels=protein_names, fontsize=24)
# plt.show()
# plt.savefig(f"/home/gluetown/brain/data/feature_attribution/inputxgrad/sheep_feature_attribution.png")

# ### pool across features (to see sequence attribution)
# pooled_feature = attribution.mean(dim=2).detach().numpy()
# plt.figure(figsize=(30, 15))
# cax = plt.imshow(pooled_feature, aspect="auto", cmap="rocket_r", interpolation='nearest')
# plt.colorbar(cax, label="Value")
# plt.xlabel("Sequence Position", fontsize=30)         
# plt.ylabel("Genes", fontsize=24)      
# plt.title(f"Best predicted species attribution over sequence", fontsize=24)
# plt.grid(visible=False)
# plt.xticks(range(pooled_feature.shape[1]), fontsize=24)
# plt.yticks(range(pooled_feature.shape[0]), labels=protein_names, fontsize=24)
# plt.show()
# plt.savefig(f"/home/gluetown/brain/data/feature_attribution/inputxgrad/sheep_seq_attribution.png")
    



# plt.figure(figsize=(12, 6))
# plt.figure(figsize=(10, 6))
# cax = plt.imshow(padded_arrays, aspect="auto", cmap="rocket_r", interpolation='nearest')
# plt.colorbar(cax, label="Value")  
# plt.xlabel("Sequence Position")         
# plt.ylabel("Genes")      
# plt.title(f"Human_attribution_values")
# plt.grid(visible=False)
# plt.xticks(range(max_length))
# plt.yticks(range(len(padded_arrays)), labels=protein_names[0:len(padded_arrays)])
# plt.show()
# plt.savefig(f"/home/gluetown/brain/data/feature_attribution/inputxgrad/human_.png")
    

# batched_data = batch_list(data)
# all_attribution = []
# for i in range(320):
#     feature_attribution = []
#     for batch in batched_data:
#         log_memory_usage()
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

# filehandler = open(b"/home/gluetown/brain/data/feature_attribution/inputxgrad/all_attribution.pkl","wb")
# pkl.dump(all_attribution,filehandler)

# protein_names = [i[0] for i in data]
# for ind, feature_attribution_data in enumerate(all_attribution):
#     # tokens = [i for i in sample[1]]  # Replace with real token strings
#     # attribution_wo_padding = aggregated_attributions[0][0:len(sample[1])]
#     plt.figure(figsize=(12, 6))
#     max_length = max(len(arr) for arr in feature_attribution_data)
#     padded_arrays = np.array([
#         np.pad(arr, (0, max_length - len(arr)), constant_values=np.nan) for arr in feature_attribution_data
#     ])
#     plt.figure(figsize=(10, 6))
#     cax = plt.imshow(padded_arrays, aspect="auto", cmap="rocket_r", interpolation='nearest')
#     plt.colorbar(cax, label="Value")  
#     plt.xlabel("Sequence Position")         
#     plt.ylabel("Genes")      
#     plt.title(f"Feature{ind}_attribution_values")
#     plt.grid(visible=False)
#     plt.xticks(range(max_length))
#     plt.yticks(range(len(padded_arrays)), labels=protein_names[0:len(padded_arrays)])
#     plt.show()
#     plt.savefig(f"/home/gluetown/brain/data/feature_attribution/inputxgrad/human_feature{ind}.png")
        
    
    
    
#     # plt.scatter(range(len(tokens)), attribution_wo_padding, color="blue", s=1, alpha=0.7)
#     # plt.xticks(ticks=range(len(tokens)), labels=tokens)
#     # plt.title(f"{sample[0]}Aggregated Attributions per Token")
#     # plt.xlabel("Tokens")
#     # plt.ylabel("Attribution Score")
#     # plt.grid(axis="y", linestyle="--", alpha=0.5)
#     # plt.show()
#     # plt.savefig(f"/home/gluetown/brain/data/feature_attribution/inputxgrad/human_test_{sample[0]}.png")




### BRAIN CLUSTERS ### 
model_file = "/home/gluetown/brain/data/embeddings/test/mmseqs/brain_clusters/all_brain_gys_dryad/gys_dryad_kfold_epoch_50_num_layers1_hidden_dim_256_seed42_early_stopping_dropout/model/1_1_model.pth"
seed = 45
set_seed(seed)
log_memory_usage()
input_dim = 320
output_dim = 1
batch_size = 10
learning_rate = 1e-4
num_folds = 5
max_epochs = 50
use_early_stopping = True
hidden_dim = 256
use_dropout = True
num_layers = 1
point_net_model = PointNetRegHead2(first_dim=input_dim, global_features = hidden_dim, k = output_dim, num_layers = num_layers, use_dropout=use_dropout)
point_net_model.load_state_dict(torch.load(model_file, weights_only=True, map_location=torch.device('cpu')))
print("Model loaded successfully with matching parameters.")
print("Model's state_dict:")
for param_tensor in point_net_model.state_dict():
    print(param_tensor, "\t", point_net_model.state_dict()[param_tensor].size())

model_name = "esm2_t6_8M_UR50D"
model_data, regression_data = _download_model_and_regression_data(model_name)
cfg = model_data["cfg"]["model"]
state_dict = model_data["model"]
state_dict = upgrade_state_dict(state_dict)
alphabet = esm.data.Alphabet.from_architecture("ESM-1b")
esm_model = esm2.ESM2(
    num_layers=cfg.encoder_layers,
    embed_dim=cfg.encoder_embed_dim,
    attention_heads=cfg.encoder_attention_heads,
    alphabet=alphabet,
    token_dropout=False,
)
stacked_model = StackedEsmPointnet(model_data, model_file, first_dim=input_dim, global_features = hidden_dim, k = output_dim,use_dropout=use_dropout)
log_memory_usage()


# esm model wrappers
def forward_wrapper(batch_tokens):
    embeddings = model.embed_tokens(batch_tokens)
    embeddings = embeddings.detach()
    embeddings.requires_grad = True
    outputs = model(embeddings, batch_tokens, repr_layers=[6], return_contacts=True)
    return outputs["representations"][6][0, :, :]

def forward_wrapper(embeddings):
    outputs = model(embeddings, batch_tokens, repr_layers=[6], return_contacts=True)
    return outputs["representations"][6]


# stacked model wrapper
def forward_wrapper(embeddings):
    outputs = stacked_model(embeddings, batch_tokens)
    return outputs

# # point net
# def forward_wrapper(embeddings):
#     outputs = point_net_model(embeddings)
#     return outputs



# Load ESM-2 model
# model, alphabet = esm.pretrained.esm2_t6_8M_UR50D() # load_model_and_alphabet_hub("esm2_t6_8M_UR50D")
batch_converter = alphabet.get_batch_converter()
model.eval()  # disables dropout for deterministic results
dryad_file = "/home/gluetown/brain/data/embeddings/h5_files/dryad_reg_out_order.h5"
pad_x1, y1, order_labels1, vocab1, mask1, species_labels1, protein_labels1  = load_output(dryad_file, protein_labels = True, order = True, indiv_datasets = True)
x_list1, y_list1, y_list_normalized1, y_list_standardized1, order_list1, mask1, species_labels1 =  initialize_dataset(pad_x1, y1, order_labels1, vocab1, mask1, species_labels1)
gys_file = "/home/gluetown/brain/data/embeddings/h5_files/gys_reg_out_order.h5"
pad_x2, y2, order_labels2, vocab2, mask2, species_labels2, protein_labels2  = load_output(gys_file, protein_labels = True, order = True, indiv_datasets = True)        
x_list2, y_list2, y_list_normalized2, y_list_standardized2, order_list2, mask2, species_labels2 =  initialize_dataset(pad_x2, y2, order_labels2, vocab2, mask2, species_labels2)
x_list = x_list1 + x_list2
y_list  = y_list1 + y_list2
order_list = order_list1 + order_list2
species_labels = species_labels1 + species_labels2
protein_labels1.update(protein_labels2)
protein_labels = protein_labels1
all_brain_genes = pd.read_csv("/group/gquongrp/workspaces/claireh/brain/data/weak_sup/overlap_brain_genes.csv", header=None)[0].tolist()
cluster_indices = match_dict_indices(protein_labels, all_brain_genes)
new_x_list, new_y_list, new_order_list, new_species_labels, new_protein_labels = subset_data(cluster_indices, x_list, y_list, order_list, protein_labels, species_labels)



### human test set
human_idx = new_species_labels.index("UP000005640_9606.fasta")
human_x = new_x_list[human_idx] # 187

check_dataloader = create_dataloader(new_x_list, new_y_list, new_species_labels, 1, is_train=False)
actual, pred = get_predictions(point_net_model, check_dataloader)
common_names = pd.read_csv("/home/gluetown/brain/data/common_names.csv").set_index("Proteome_ID")["Common Name"].to_dict()
predicted = {common_names[s.split("_")[0]]:{} for s in new_species_labels}
for ind, s in enumerate(new_species_labels):
    predicted[common_names[s.split("_")[0]]]["pred"] = pred[ind][0]
    predicted[common_names[s.split("_")[0]]]["actual"] = actual[ind]
    predicted[common_names[s.split("_")[0]]]["mse"] = mean_squared_error(pred[ind], [actual[ind]])
    predicted[common_names[s.split("_")[0]]]["species_id"] = s
# predicted[s] = mean_squared_error(pred[ind], [actual[ind]])
min_mse_species = min(predicted, key=lambda s: predicted[s]['mse'])
best_species = sorted(predicted.items(), key=lambda item: item[1]['mse'])[:10]

# oranguatans
# UP000001595_9601.fasta
orangutan_idx = new_species_labels.index("UP000001595_9601.fasta")
orangutan_x = new_x_list[orangutan_idx]
orangutan_proteins = new_protein_labels["UP000001595_9601.fasta"]
orangutan_fasta = "/home/gluetown/brain/data/uniprot_files/gys/UP000001595_9601.fasta"


sheep_idx = new_species_labels.index("UP000002356_9940.fasta")
sheep_x = new_x_list[sheep_idx]
sheep_proteins = new_protein_labels["UP000002356_9940.fasta"]
sheep_fasta = "/home/gluetown/brain/data/uniprot_files/gys/UP000002356_9940.fasta"

rat_fasta = "/home/gluetown/brain/data/uniprot_files/gys/UP000002494_10116.fasta"
rat_proteins = new_protein_labels["UP000002494_10116.fasta"]

human_fasta = "/home/gluetown/brain/data/uniprot_files/gys/UP000005640_9606.fasta"
human_proteins = new_protein_labels["UP000005640_9606.fasta"]

chicken_fasta = "/home/gluetown/brain/data/uniprot_files/tsuboi/UP000000539_9031.fasta"
chicken_proteins = new_protein_labels["UP000000539_9031.fasta"]

chimp_fasta = "/home/gluetown/brain/data/uniprot_files/gys/UP000002277_9598.fasta"
chimp_proteins = new_protein_labels["UP000002277_9598.fasta"]

data, protein_names = intialize_attribution_data(chicken_fasta, chicken_proteins)
output_file = "chicken"


batch_labels, batch_strs, batch_tokens = batch_converter(data)
batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
embeddings = model.embed_tokens(batch_tokens)
log_memory_usage()
input_x_gradient = InputXGradient(forward_wrapper)
attribution = input_x_gradient.attribute(embeddings)
# out = stacked_model(embeddings, batch_tokens)

with open(f"/home/gluetown/brain/data/feature_attribution/inputxgrad/{output_file}_stacked_attribution.pkl", "wb") as filehandler:
    pkl.dump(attribution, filehandler)


with open(b"/home/gluetown/brain/data/feature_attribution/inputxgrad/chicken_stacked_attribution.pkl","rb") as f:
    attribution = pkl.load(f)

# PLOTS

### pool across sequence (to see feature attribution)
pooled_seq = attribution.mean(dim=1).detach().numpy()
plt.figure(figsize=(30, 15))
cax = plt.imshow(pooled_seq, aspect="auto", cmap="mako", interpolation='nearest')
cbar = plt.colorbar(cax, label="Value")
cbar.ax.set_ylabel("Value", fontsize=24)
cbar.ax.tick_params(labelsize=36)  
plt.xlabel("Features", fontsize=36)         
plt.ylabel("Genes", fontsize=36)      
plt.grid(visible=False)
plt.xticks(range(pooled_seq.shape[1]), fontsize=36)
plt.yticks(range(pooled_seq.shape[0]), labels=protein_names, fontsize=36)
plt.show()
plt.savefig(f"/home/gluetown/brain/data/feature_attribution/inputxgrad/{output_file}_feature_attribution.png")

### pool across features (to see sequence attribution)
pooled_feature = attribution.mean(dim=2).detach().numpy()
masked_pooled_feature = np.ma.array(pooled_feature, mask=np.ones_like(pooled_feature))
max_len = pooled_feature.shape[1]
for i, batch_len in enumerate(batch_lens):
    masked_pooled_feature.mask[i, :batch_len] = False

plt.figure(figsize=(30, 15))
cax = plt.imshow(masked_pooled_feature, aspect="auto", cmap="mako", interpolation='nearest')
plt.clim(pooled_feature.min(), pooled_feature.max())  # Ensure color scaling uses full data range
cbar = plt.colorbar(cax, label="Value")
cbar.ax.tick_params(labelsize=36)  
plt.xlabel("Sequence Position", fontsize=36)         
plt.ylabel("Genes", fontsize=36)      
plt.grid(visible=False)
plt.xticks(range(pooled_feature.shape[1]), fontsize=36)
plt.yticks(range(pooled_feature.shape[0]), labels=protein_names, fontsize=36)
plt.tight_layout()
plt.show()
plt.savefig(f"/home/gluetown/brain/data/feature_attribution/inputxgrad/{output_file}_seq_attribution.png")






# prot = "Q5R590"
# prot_seq_attribution = pooled_feature[protein_names.index(prot)]
# prot_seq = data[[i[0] for i in data].index(prot)][1]
# indices = np.where(abs(prot_seq_attribution) > np.quantile(abs(prot_seq_attribution), 0.95))[0].tolist()
# indices

prot = "P53449"
prot_seq_attribution = pooled_feature[protein_names.index(prot)]
prot_seq = data[[i[0] for i in data].index(prot)][1]
indices = np.where(prot_seq_attribution > 1e-5)[0].tolist()
indices = np.where(abs(prot_seq_attribution) > np.quantile(abs(prot_seq_attribution), 0.95))[0].tolist()
indices
