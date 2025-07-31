import os
import torch
import regex as re
import h5py
import pandas as pd
from tqdm import tqdm
import time

# nested tensor version
def find_length_of_nested_tensor(sequences):
    length = 0
    for s in sequences:
        length += 1
    return length

def save_output(x, y, protein_labels, mask, output_file):
    with h5py.File(output_file, 'w') as f:
        f.create_dataset('x', data=x)
        f.create_dataset('y', data=y)
        # f.create_dataset('protein_labels', data=protein_labels)
        f.create_dataset('mask', data=mask)
        group = f.create_group('protein_labels')
        for i, sublist in enumerate(protein_labels):
            dt = h5py.special_dtype(vlen=str)
            group.create_dataset(f'sublist_{i}', data=sublist, dtype=dt)
    print(f"Saved output to {output_file}")


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
    return species_embeddings, protein_labels

def pad_embeddings(x, device="cpu"):
    nt = torch.nested.nested_tensor(x, 
                                dtype=torch.float, 
                                device=device)
    padded_x = torch.nested.to_padded_tensor(nt, padding=0.0)    
    num_samples = find_length_of_nested_tensor(padded_x)
    max_length = len(padded_x[0])

    mask = torch.zeros((num_samples, max_length), dtype=torch.bool, device=device)
    for i, proteins in enumerate(x):
        if len(proteins) < max_length:
            for j in range(len(proteins), max_length):
                mask[i, j] = True
    return padded_x, mask

if __name__ == "__main__":
    time1 = time.time()
    meta_df = pd.read_csv("/home/gluetown/brain/data/metadata.csv.gz", compression = "gzip")
    #### Real data
    input_dir = "/home/gluetown/brain/final_embeddings/"
    output_dir = "/home/gluetown/brain/outputs/encoder_cls_2/"
    #### Test set
    # input_dir = "/home/gluetown/brain/test_set/test/embeddings/"
    # output_dir="/home/gluetown/brain/test_set/test/"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file= output_dir + "embeddings.h5"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = []
    y = []
    species_id = []
    all_protein_labels = []

    for species in os.listdir(input_dir):
        print(f"Starting {species}")
        species_embeddings, protein_labels = initialize_input(f"{input_dir}{species}/")
        print((len(species_embeddings), len(species_embeddings[0])))
        x.append(species_embeddings)
        all_protein_labels.append(protein_labels)
        id = re.findall(r"(UP.*.fasta)", species)[0]
        species_id.append(id) 
        brainsize = meta_df.loc[meta_df["species_id"] == id]['Brain.resid'].iloc[0]
        y.append(brainsize)
        print(f"Finished {species}, {len(species_embeddings)} tensors")
    y = torch.tensor(y, dtype=torch.float, device=device)

    # Padding
    print(f"Number of samples: {len(x)}")
    for i in range(len(x)): 
        print(f"Shape at index {i} before padding: {len(x[i])}")
    padded_embeddings, mask = pad_embeddings(x, device)
    save_output(padded_embeddings, y, all_protein_labels, mask, output_file)
    print(f"Shape after padding: {padded_embeddings.shape}")
    time2 = time.time()
    print(f"Total time: {time2 - time1}")