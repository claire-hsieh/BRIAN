import os
import pandas as pd
import subprocess
import torch
import numpy as np
import h5py as h5
from tqdm import tqdm
import pickle as pkl
import statsmodels.api as sm
import statsmodels.formula.api as smf
import itertools


def walk_directory(parent_dir, output_file):
    # create file with uniprot ids and species ids
    result = {'uniprot_id': [], 'species_id': []}
    up_files = os.listdir(parent_dir)
    for up_file in up_files:
        ids = subprocess.run(["grep", '>', f"{parent_dir}{up_file}", "| cut -f 2 -d'|' "], 
                            capture_output=True, 
                            text=True)
        ids = ids.stdout.split("\n")
        for id in ids:
            try: 
                result['uniprot_id'].append(id.split('|')[1])
                result['species_id'].append(up_file.split("/")[-1])
            except:
                continue
        print(f"Read in {len(result['uniprot_id'])} proteins from {up_file}")    
    df = pd.DataFrame(result)
    df.to_csv(output_file, index=False, mode='a', header=not os.path.exists(output_file))
    
def match_dataset_to_proteome(brain_input_file = "/home/gluetown/brain/data/metadata_raw.csv", uniprot_input_file = "/home/gluetown/brain/data/uniprot_table.txt.gz", output_file = "/home/gluetown/brain/data/common_names.csv"):
    # Matching brain size dataset to proteome
    # using dataset species names and uniprot species names (are usually slightly different)
    brain_df = pd.read_csv(brain_input_file)
    uniprot_df = pd.read_csv(uniprot_input_file, compression = 'gzip', sep = '\t')
    uniprot_df = uniprot_df[uniprot_df["SUPERREGNUM"] == "eukaryota"]
    extra = {i.split("_")[0]:"" for i in os.listdir("../data/extra/")}
    common_names = {}
    matched_species = {}
    print("Species (brain data) -> Species (UdniProt)")
    uniprot_species = uniprot_df["Species Name"].str.lower().tolist()
    for species in brain_df['binomial'].unique():
        species_lower = species.replace("_", " ").lower()
        for up_species in uniprot_species:
            if species_lower in up_species:
                print(f"{species_lower} -> {up_species} -> {uniprot_df.loc[uniprot_df['Species Name'].str.lower() == up_species]['Proteome_ID'].values[0]}")
                up_id = uniprot_df.loc[uniprot_df['Species Name'].str.lower() == up_species]['Proteome_ID'].values[0]
                species_id = uniprot_df.loc[uniprot_df["Species Name"].str.lower() == up_species]["Proteome_ID"].values[0]
                matched_species[species_id] = species_lower
                common_names[species_id] = up_species
    common = pd.DataFrame.from_dict(common_names, orient='index').reset_index()
    matched = pd.DataFrame.from_dict(matched_species, orient='index').reset_index()
    common = common.merge(matched, on='index')
    common.columns = ['Proteome_ID', 'Common Name', 'Species']
    common.to_csv(output_file, index=False)

def initialize_metadata():
    # get raw data from all datasets (done in R)
    # attach uniprot_ids and proteins
    output_file = "/home/gluetown/brain/data/all_species_protein_ids.csv"  
    parent_dir = "/home/gluetown/brain/data/uniprot_files/uniprot/"  
    walk_directory(parent_dir, output_file)
    parent_dir = "/home/gluetown/brain/data/uniprot_files/tsuboi_uniprot/"  
    walk_directory(parent_dir, output_file)
    parent_dir = "/home/gluetown/brain/data/uniprot_files/dryad_uniprot/"  
    walk_directory(parent_dir, output_file)
    metadata_df = pd.read_csv("/home/gluetown/brain/data/metadata_raw.csv")
    uniprot_df = pd.read_csv("../data/uniprot_table.txt.gz", compression = 'gzip', sep = '\t')
    uniprot_df = uniprot_df[uniprot_df["SUPERREGNUM"] == "eukaryota"]
    embedded_species = [i.replace("_", " ") for i in metadata_df["binomial"].unique().tolist()]
    # match_dataset_to_proteome()
    # merge files
    species_id_df = pd.read_csv("/home/gluetown/brain/data/all_species_protein_ids.csv")
    common = pd.read_csv("/home/gluetown/brain/data/common_names.csv")
    metadata_df["binomial"] = metadata_df["binomial"].str.lower()
    metadata_df = metadata_df.merge(common, left_on="binomial", right_on="Species")
    species_id_df["Proteome_ID"] = species_id_df["species_id"].str.split('_').str[0]
    species_id_df = pd.merge(species_id_df, common, on='Proteome_ID', how='inner')
    species_id_df = pd.merge(species_id_df, metadata_df, on="Proteome_ID", how='inner')
    species_id_df.drop('Common Name_y',axis=1, inplace=True)
    species_id_df.drop('Species_y',axis=1, inplace=True)
    species_id_df = species_id_df.rename(columns={'Common Name_x': 'Common Name'})
    species_id_df = species_id_df.rename(columns={'Species_x': 'Species'})
    species_id_df.to_csv("/home/gluetown/brain/data/metadata_all_2.csv.gz", compression = "gzip")
    return species_id_df

def intialize_embeddings(output_file, metadata, directory = "/group/gquongrp/workspaces/claireh/brain/embeddings/tsuboi_embeddings/", data_dict_file=None):
    if data_dict_file == None:
        data_dict = {species: {} for species in os.listdir(directory)}
        for species in tqdm(os.listdir(directory)):
            print(species)
            for root, dirs, files in os.walk(os.path.join(directory, species)):
                for file in files:
                    if file.endswith(".pt"):
                        try:
                            gene_id = file.split("|")[1]
                            species_id = species
                            data_dict[species_id][gene_id] = torch.load(f"{root}/{file}")["mean_representations"][6]
                        except:
                            print(file)
        output_file_path = f"{output_file.rsplit('/', 1)[0]}/data_dict.pkl"
        with open(output_file_path, 'wb') as f:
            pkl.dump(data_dict, f)
        data_dict = {species: data_dict[species] for species in data_dict if data_dict[species]}
    else:
        with open(data_dict_file, "rb") as f:
            data_dict = pkl.load(f)
    embeddings = {species: np.array(list(data_dict[species].values())) for species in data_dict.keys()}
    labels = {}
    order = {}
    normalized_labels = {}
    standardized_labels = {}
    for species in embeddings.keys():
        try:
            species_tmp = metadata.loc[metadata["species_id"] == species]
            species_tmp_filtered = species_tmp.loc[species_tmp["uniprot_id"].isin(data_dict[species].keys())]
            labels[species] = species_tmp_filtered["residuals"].values.tolist()[0]
            order[species] = species_tmp_filtered["order"].values.tolist()[0]
            # normalized_labels[species] = species_tmp_filtered["normalized_residuals"].values.tolist()[0]
            # standardized_labels[species] = species_tmp_filtered["standardized_residuals"].values.tolist()[0]
        except:
            print(f"Error with {species}")
    # Pad the sequences to the same length
    max_length = max(len(v) for v in embeddings.values())
    padded_embeddings = {k: np.pad(v, ((0, max_length - len(v)), (0, 0)), mode='constant') for k, v in embeddings.items()}
    masks = {k: np.pad(np.ones(len(v)), (0, max_length - len(v)), mode='constant') for k, v in embeddings.items()}
    embeddings_array = np.array(list(padded_embeddings.values()))
    labels_array = np.array(list(labels.values()))
    masks_array = np.array(list(masks.values()))
    protein_labels = {species: list(data_dict[species].keys()) for species in data_dict.keys()}
    species_labels = list(data_dict.keys())
    protein_labels_str = [f"{species}:{gene}" for species in protein_labels for gene in protein_labels[species]]
    order_labels = list(order.values())
    # normalized_labels = np.array(list(normalized_labels.values()))
    # standardized_labels = np.array(list(standardized_labels.values()))
    # Write to HDF5 file
    with h5.File(output_file, "w") as f:
        f.create_dataset("x", data=embeddings_array)
        f.create_dataset("y", data=labels_array)
        f.create_dataset("mask", data=masks_array)
        f.create_dataset("protein_labels", data=np.array(protein_labels_str, dtype='S'))
        f.create_dataset("species_labels", data=np.array(species_labels, dtype='S'))
        f.create_dataset("order_labels", data=np.array(order_labels, dtype='S'))
        # f.create_dataset("normalized_labels", data=np.array(normalized_labels))
        # f.create_dataset("standardized_labels", data=np.array(standardized_labels))

def load_output(input_file, protein_labels = False, order = False):
    with h5.File(input_file, 'r') as f:
        x = torch.tensor(f['x'][:], dtype=torch.float32)
        y = torch.tensor(f['y'][:], dtype=torch.float32)
        mask = torch.tensor(f['mask'][:], dtype=torch.float32)
        species_labels = [s.decode('utf-8') for s in f["species_labels"][:]]     
        # normalized_labels = torch.tensor(np.array([s.decode('utf-8') for s in f["normalized_labels"][:]], dtype=float))
        # standardized_labels = torch.tensor(np.array([s.decode('utf-8') for s in f["standardized_labels"][:]], dtype=float))
        print(f"Loaded data from {input_file}")
        if protein_labels and order:
            protein_labels = f['protein_labels'][:]
            if isinstance(protein_labels[0], bytes):
                protein_labels = [label.decode('utf-8') for label in protein_labels]
            order_labels = f['order_labels'][:]
            if isinstance(order_labels[0], bytes):
                order_labels = [label.decode('utf-8') for label in order_labels]
            vocab = {label: i for i, label in enumerate(set(order_labels))}
            order_labels = [vocab[label] for label in order_labels]
            return x, y, order_labels, vocab, mask, species_labels, protein_labels
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

def get_indices(element, lst):
    return [i for i in range(len(lst)) if lst[i] == element]

def extract_data_by_source(metadata, order_dict, dataset, residuals_dataset, pad_x, protein_labels, mask, species_labels, output_file="/home/gluetown/brain/data/embeddings/h5_files/gys.h5"):
    # Seperates out datasets
    with open(residuals_dataset, "rb") as f:
        residual_species = pkl.load(f)
    protein_label_df = pd.DataFrame([item.split(':') for item in protein_labels], columns=['species_id', 'uniprot_id'])
    new_protein_labels = {}
    for tmp in protein_label_df.groupby("species_id"):
        new_protein_labels[tmp[0]] = list(tmp[1]["uniprot_id"].values)
    with open(output_file.split(".h5")[0] + ".pkl", "wb") as f2:
        pkl.dump(new_protein_labels, f2)
    trouble_makers = ["UP000189704_1868482.fasta", "UP000009136_9913.fasta", "UP000694520_30521.fasta"]
    species_ids = metadata.loc[metadata["source"] == dataset]["species_id"]
    species_indices = [get_indices(i, species_labels) for i in list(set(species_ids.values)) if i not in trouble_makers]
    species_indices = list(itertools.chain(*species_indices))
    new_x = pad_x[species_indices]
    new_species_labels = [species_labels[i] for i in species_indices]
    new_order_labels = [order_dict[i] for i in new_species_labels]
    new_mask = mask[species_indices]
    # new_protein_labels = pd.DataFrame.from_dict(new_protein_labels, orient='index').reset_index()
    new_y = [residual_species[s] for s  in new_species_labels ]
    with h5.File(output_file, "w") as f:
        f.create_dataset("x", data=new_x)
        f.create_dataset("y", data=new_y)
        f.create_dataset("mask", data=new_mask)
        # f.create_dataset("protein_labels", data=new_protein_labels)
        f.create_dataset("species_labels", data=np.array(new_species_labels, dtype='S'))
        f.create_dataset("order_labels", data=np.array(new_order_labels, dtype='S'))
          
def calculate_residuals(df):
    model = smf.ols('log_brain_mass ~ log_body_mass', data=df).fit()    
    coefficients = model.params
    a = coefficients[0]
    b = coefficients[1]
    print(f"a: {a}")
    print(f"b: {b}")
    df['residuals'] = model.resid
    return df


if __name__ == "__main__":
    # already done
    # initialize_metadata()

    # create embeddings
    output_file = "/home/gluetown/brain/data/embeddings/h5_files/all_3.h5"
    metadata_df = pd.read_csv("/home/gluetown/brain/data/metadata_all_2.csv.gz", compression="gzip")
    directory = "/group/gquongrp/workspaces/claireh/brain/data/embeddings/final_embeddings/"
    data_dict_file = "/home/gluetown/brain/data/embeddings/h5_files/data_dict.pkl"
    # intialize_embeddings(output_file, metadata_df, directory, data_dict_file)


    # Create new files for each dataset
    input_file = "/home/gluetown/brain/data/embeddings/h5_files/all_3.h5"
    pad_x, y, order_labels, vocab, mask, species_labels  = load_output(input_file, order=True)

    metadata_df = pd.read_csv("/home/gluetown/brain/data/metadata_all_2.csv.gz", compression = "gzip")
    with h5.File(input_file, "r") as f:
        protein_labels = np.array([s.decode('utf-8') for s in f["protein_labels"][:]])

# for dataset in ["gys", "dryad", "tsuboi"]:
#     tmp_df = metadata_df.loc[metadata_df["source"] == dataset].drop_duplicates(subset="species_id")
#     tmp_df = calculate_residuals(tmp_df) 
#     residual_species = tmp_df.set_index("species_id")["reg_out_residuals"].to_dict()
#     with open(f"/home/gluetown/brain/data/embeddings/h5_files/{dataset}_reg_out_order_residuals.pkl", "wb") as f:
#         pkl.dump(residual_species, f)

# trouble_makers = ["UP000189704_1868482.fasta", "UP000009136_9913.fasta", "UP000694520_30521.fasta"]
# order_dict = metadata.set_index("species_id")["order"].to_dict()

# metadata = pd.read_csv("/home/gluetown/brain/data/metadata_all_2.csv.gz", compression = "gzip")
# input_file = "/home/gluetown/brain/data/embeddings/h5_files/all_3.h5"
# pad_x, y, order_labels, vocab, mask, species_labels, protein_labels  = load_output(input_file, protein_labels = True, order = True)
# order_dict = metadata.set_index("species_id")["order"].to_dict()
# extract_data_by_source(metadata, order_dict, "gys", "/home/gluetown/brain/data/embeddings/h5_files/gys.pkl", pad_x, protein_labels, mask, species_labels, output_file="/home/gluetown/brain/data/embeddings/h5_files/gys.h5")
# extract_data_by_source(metadata, order_dict, "dryad", "/home/gluetown/brain/data/embeddings/h5_files/dryad.pkl", pad_x, protein_labels, mask, species_labels, output_file="/home/gluetown/brain/data/embeddings/h5_files/dryad.h5")
# extract_data_by_source(metadata, order_dict, "tsuboi", "/home/gluetown/brain/data/embeddings/h5_files/tsuboi.pkl", pad_x, protein_labels, mask, species_labels,  output_file="/home/gluetown/brain/data/embeddings/h5_files/tsuboi.h5")
