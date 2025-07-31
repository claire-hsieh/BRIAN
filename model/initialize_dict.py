import torch
from tqdm import tqdm
import os
import pandas as pd
import regex as re
import time

def initialize_input(input_dir):
    species_embeddings = []
    protein_labels = []
    for root, dirs, files in os.walk(input_dir, topdown=True):
        for file in tqdm(files):
            if file.endswith(".pt"):
                try:
                    embedding = torch.load(root + "/" + file)
                    species_embeddings.append(embedding['mean_representations'][6].tolist())  # Convert tensor to list
                    protein_labels.append(embedding['label'])  # Convert tensor to list
                except:
                    print(f"Error loading {file}")
    species_embeddings = torch.tensor(species_embeddings)
    return species_embeddings, protein_labels

if __name__ == "__main__":
    ### TEST DATA ###
    # input_dir = "/home/gluetown/brain/test_set/test/embeddings/"
    # output_dir="/home/gluetown/brain/test_set/test/"
    ### REAL DATA ###
    input_dir = "/home/gluetown/brain/data/embeddings/final_embeddings/"
    output_dir = "/home/gluetown/brain/outputs/"
    
    meta_df = pd.read_csv("/home/gluetown/brain/data/metadata.csv.gz", compression = "gzip")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file= output_dir + "embeddings.h5"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = {}
    y = {}
    species_id = []
    protein_labels = {species:[] for species in os.listdir(input_dir)}

    for species in os.listdir(input_dir):
        print(f"Starting {species}")
        species_embeddings, prot_label = initialize_input(f"{input_dir}{species}/")
        print((len(species_embeddings), len(species_embeddings[0])))
        x[species] = species_embeddings
        protein_labels[species].append(protein_labels)
        id = re.findall(r"(UP.*.fasta)", species)[0]
        species_id.append(id) 
        brainsize = meta_df.loc[meta_df["species_id"] == id]['Brain.resid'].iloc[0]
        y[species] = brainsize
        print(f"Finished {species}, {len(species_embeddings)} tensors")
    # y = torch.tensor(y, dtype=torch.float, device=device)
    torch.save(x, output_dir + "embeddings.pt")
    torch.save(y, output_dir + "labels.pt")
    torch.save(protein_labels, output_dir + "protein_labels.pt")


    ### TO LOAD DATA ###
    # x = torch.load(output_dir + "embeddings.pt")
    # y = torch.load(output_dir + "labels.pt")
    # protein_labels = torch.load(output_dir + "protein_labels.pt")

    # x.keys(), y.keys(), protein_labels.keys()
    # to load y as a tensor:
    # y = torch.tensor(list(y.values()))