from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import os
from bs4 import BeautifulSoup
import requests
import shutil
import regex as re
from pathlib import Path
import json
import torch
import tqdm
import argparse
import requests
from bs4 import BeautifulSoup
import regex as re
import pandas as pd
import os
from tqdm import tqdm
import os
import pandas as pd
import subprocess

# uniprot functions
def match_dataset_to_proteome(pheno_df, pheno_col, output_file, uniprot_df):
    # Matching brain size dataset to proteome
    # using dataset species names and uniprot species names (are usually slightly different)
    uniprot_df = uniprot_df[uniprot_df["SUPERREGNUM"] == "eukaryota"]
    common_names = {}
    matched_species = {}
    print("Species (brain data) -> Species (UniProt)")
    uniprot_species = uniprot_df["Species Name"].str.lower().tolist()
    for species in pheno_df[pheno_col].unique():
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

def get_links(prot_id, output_dir ):       
    base_url = "https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/reference_proteomes/Eukaryota/"

    # print(len(prot_id))
    
    for i in prot_id:
        url = f"{base_url}/{i}"
        print(url)
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        fasta_links = [url + "/" + node.get('href') for node in soup.find_all('a') if re.search(r'\d+\.fasta\.gz$', node.get('href'))]
        for link in fasta_links:
            print(link)
            filename = output_dir + link.split('/')[-1]
        if len(fasta_links) == 0:
            print(f"No fasta files found for {j}")
        else:
            os.system(f"wget {link} -O {filename}")

def regress_out(data, target_col, var_to_regress_out):
    mask = ~(data[var_to_regress_out].isna() | data[target_col].isna())
    X = np.array(data[var_to_regress_out][mask]).reshape(-1, 1)
    y = data[target_col][mask]
    model = LinearRegression()
    model.fit(X, y)
    residuals = pd.Series(index=data.index, dtype=float)
    residuals[mask] = y - model.predict(X)
    return residuals

def process_esm_embeddings(input_folder, output_file, protein_file):
    # input_folder contains species folders with ESM embeddings
    all_embeddings = {}
    protein_names = {i:[] for i in os.listdir(input_folder)}
    for species_dir in tqdm(os.listdir(input_folder)):
        print(species_dir)
        embeddings = []
        if os.path.isdir(os.path.join(input_folder, species_dir)):
            for embedding_file in tqdm(os.listdir(os.path.join(input_folder, species_dir))):
                if embedding_file.endswith(".pt"):
                    try:
                        embedding = torch.load(os.path.join(input_folder, species_dir, embedding_file))["mean_representations"][6]
                        embeddings.append(embedding)
                        protein_names[species_dir].append(embedding_file.split("|")[1])
                    except:
                        print(f"Error with {species_dir}/{embedding_file}")
            all_embeddings[species_dir] = torch.stack(embeddings)
    torch.save(all_embeddings, output_file)
    with open(protein_file, "w") as f:
        json.dump(protein_names, f)


# metadata functions
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
    
def initialize_metadata(brainsize_df, uniprot_df, output_file):
    output_file = "data/all_species_protein_ids.csv"  
    parent_dir = "data/uniprot/"  
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

def copy_items_to_dir(item_list, output_dir, verbose=True):
    """
    Copy all items in list to output directory

    Parameters:
    item_list : list of paths to copy
    output_dir : destination directory
    verbose : print progress
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    copied = []
    failed = []
    for item in item_list:
        try:
            src_path = Path(item)
            dst_path = output_path / src_path.name
            if src_path.is_file():
                shutil.copy2(src_path, dst_path)
            elif src_path.is_dir():
                shutil.copytree(src_path, dst_path)
            copied.append(item)
            if verbose:
                print(f"Copied: {item}")
        except Exception as e:
            failed.append((item, str(e)))
            if verbose:
                print(f"Failed to copy {item}: {e}")
    if verbose:
        print(f"\nCopy complete:")
        print(f"Successfully copied: {len(copied)}")
        print(f"Failed: {len(failed)}")
        if failed:
            print("\nFailed items:")
            for item, error in failed:
                print(f"{item}: {error}")
    return copied, failed

def uniprot_species(uniprot_dir, output_file): 
    # create file with uniprot ids (proteins, ie. A0A1U7QTL4) and species ids (UP000...)
    uniprot_files = os.listdir(uniprot_dir)    
    species_uniprot = {file:[] for file in uniprot_files}
    for file in uniprot_files:
        result = subprocess.run(f"grep '>' {uniprot_dir + file} | cut -f 2 -d'|'",
                                shell=True,
                                capture_output=True,
                                text=True)
        species_uniprot[file] = result.stdout.split("\n")
        df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in species_uniprot.items()]))
        melted_df = df.melt(var_name='species_id', value_name='uniprot_id').dropna()
        melted_df.to_csv(output_file, index=False)
        print(f"Saved species and protein ids to {output_file}")
        
def match_species_brain(uniprot_df, base_dir, output_file):
    # Matching brain size dataset to proteome
    # uniprot_df: uniprot_table.txt.gz from uniprot ftp README
    # base_dir: directory with protein sequences
    # output_file: output file with proteome id, brain size species name, uniprot species name
    species_dict = {i.split("_")[0]:"" for i in os.listdir(base_dir)}
    common_names = {}
    matched_species = {}
    print("Species (brain data) -> Species (UniProt)")
    uniprot_species = uniprot_df["Species Name"].str.lower().tolist()
    for species in brain_df['Binomial'].unique():
        species_lower = species.replace("_", " ").lower()
        for up_species in uniprot_species:
            if species_lower in up_species:
                # print(f"{species_lower} -> {up_species} -> {uniprot_df.loc[uniprot_df['Species Name'].str.lower() == up_species]['Proteome_ID'].values[0]}")
                up_id = uniprot_df.loc[uniprot_df['Species Name'].str.lower() == up_species]['Proteome_ID'].values[0]
                if up_id in list(species_dict.keys()):
                    species_dict[up_id] = species_lower
                    print(f"{species_lower} -> {up_species} -> {uniprot_df.loc[uniprot_df['Species Name'].str.lower() == up_species]['Proteome_ID'].values[0]}")
                    print(f"Similar: {uniprot_df.loc[[species_lower in i for i in uniprot_df["Species Name"].str.lower().tolist()]]["Species Name"].values}")
                    species_id = uniprot_df.loc[uniprot_df["Species Name"].str.lower() == up_species]["Proteome_ID"].values[0]
                    matched_species[species_id] = species_lower
                    common_names[species_id] = up_species
                    common = pd.DataFrame.from_dict(common_names, orient='index').reset_index()
                    matched = pd.DataFrame.from_dict(matched_species, orient='index').reset_index()
                    common = common.merge(matched, on='index')
                    common.columns = ['Proteome_ID', 'Common Name', 'Species']
                    common.to_csv(output_file, index=False)
                    print(f"Saved common names to {output_file}")
                    
def merge_df(uniprot_species_file, uniref_file):
    brainsize_data = pd.read_csv("/home/gluetown/brain/data/gyz043_suppl_Supplement_Data.csv")
    brainsize_data['Binomial'] = brainsize_data['Binomial'].str.lower().str.replace("_", " ")
    uniprot_df = pd.read_csv("/home/gluetown/brain/data/uniprot_table.txt.gz", compression = 'gzip', sep = '\t')
    uniprot_df = uniprot_df[uniprot_df["SUPERREGNUM"] == "eukaryota"]
    common = pd.read_csv("/home/gluetown/brain/data/common_names.csv")
    up_species = pd.read_csv(uniprot_species_file)
    uniref_df = pd.read_csv(uniref_file)
    up_species["Proteome_ID"] = up_species["species_id"].str.split("_").str[0]
    merged = up_species.merge(uniprot_df[["Proteome_ID", "Species Name"]], on="Proteome_ID")
    merged = merged.merge(common, on="Proteome_ID")
    merged.rename(columns = {"Species":"Binomial"}, inplace = True)
    merged = merged.merge(brainsize_data[["Binomial", "Brain.resid", "Sex"]], on="Binomial")
    metadata = merged[['uniprot_id', 'species_id', 'Species Name', 'Binomial', 'Sex', 'order','family' , 'genus', 'Mean_brain_mass_g','Mean_body_mass_g', 'Brain.resid']]
    metadata = metadata.merge(uniref_df, on="uniprot_id")
    metadata.to_csv("/home/gluetown/brain/data/testing/metadata.csv.gz", index=False, compression = "gzip")
    print(f"Saved metadata to /home/gluetown/brain/data/metadata.csv.gz")
    
def uniprot_data(species_file, uniprot_df):
    species_name = pd.read_csv(species_file, header = None)
    species_name.columns = ['Species Name']

    prot_id = []
    for i in species_name['Species Name']:
        match = uniprot_df[uniprot_df['Species Name'].str.contains(i)]
        if len(match) > 0:
            prot_id.append(match['Proteome_ID'].values[0])
    return prot_id, uniprot_df

%cd /home/gluetown/brain/scripts/clean_code_for_github/
# 1. load in data
brain_df = pd.read_csv("data/gyz043_suppl_Supplement_Data.csv")
uniprot_df = pd.read_csv("data/uniprot_table.txt.gz", compression="gzip", delimiter="\t")
match_brain_to_uniprot_file =  "data/uniprot_overlap.csv"
output_uniprot_dir = "data/uniprot/"
output_embeddings_dir = "data/embeddings/"

# 2. get overlap
match_dataset_to_proteome(brain_df, pheno_col='Binomial', output_file=match_brain_to_uniprot_file, uniprot_df = uniprot_df)
matched_sp = pd.read_csv(match_brain_to_uniprot_file)
to_download = list(matched_sp["Proteome_ID"].values)

# 3. Get links to download from uniprot
os.makedirs(output_uniprot_dir, exist_ok=True)
get_links(to_download, output_uniprot_dir)

# Decompress all files in the data/uniprot/ directory
for file in os.listdir(output_uniprot_dir):
    if file.endswith(".gz"):
        file_path = os.path.join(output_uniprot_dir, file)
        subprocess.run(["gzip", "-d", file_path], check=True)

# 4. log and regress brain sizes
metadata_df = brain_df.copy()
metadata_df["log_brain_mass_g"], metadata_df["log_body_mass_g"] = np.log(metadata_df["Mean_brain_mass_g"]), np.log(metadata_df["Mean_body_mass_g"])
metadata_df["residuals"] = regress_out(metadata_df, "log_brain_mass_g", "log_body_mass_g")
metadata_df["Binomial"] = [i.lower() for i in metadata_df["Binomial"].to_list()]
matched_sp["Species"] = [i.replace(" ", "_") for i in matched_sp["Species"]]
metadata_df.merge(matched_sp, left_on = "Binomial", right_on = "Species")

# 5. get embeddings
# run get_esm.py
subprocess.run(["python", "get_esm.py"], check=True)
# output_dir will be data/embeddings/ in the format uniprot_id/protein_id/emb.pt

# 6. process embeddings into single file
input_folder = output_embeddings_dir
output_file = "data/embeddings.pt"
output_protein_file = "data/all_proteins.json"
process_esm_embeddings(input_folder, output_file, output_protein_file)

# 7. labels
labels = 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process ESM embeddings and related data.")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to the input directory containing ESM embeddings.")
    args = parser.parse_args()

    input_folder = args.input_dir


    # 1. Find species that we have longevity data for and are in uniprot
    longevity_file = "/home/gluetown/viral_evo/data/longevity/anage_data.txt"
    lonegvity_df = pd.read_csv(longevity_file, delimiter="\t")
    lonegvity_df = lonegvity_df[lonegvity_df['Maximum longevity (yrs)'].notna()] #4141
    matched_file =  "/home/gluetown/viral_evo/data/longevity/uniprot_overlap.csv"

    # match_dataset_to_proteome(lonegvity_df, pheno_col='Common name', output_file=matched_file, uniprot_input_file = "/home/gluetown/brain/data/uniprot_table.txt.gz")
    # manually removed a bunch of entries (dont run above line again!!)
    matched_sp = pd.read_csv(matched_file)
    matched_sp.columns = ["index", "Proteome_ID", "Uniprot Name", "Pheno Name"]
    lonegvity_df['Common name'] = lonegvity_df['Common name'].str.lower()
    lonegvity_df = lonegvity_df.rename(columns={'Common name': 'Pheno Name'})
    red_longevity_df = pd.merge(lonegvity_df[["Pheno Name", 'Maximum longevity (yrs)', 'Body mass (g)']],matched_sp, on = "Pheno Name")
    red_longevity_df["log_body_mass"] = np.log(red_longevity_df["Body mass (g)"])
    # red_longevity_df[red_longevity_df["Body mass (g)"].notna() & red_longevity_df["Maximum longevity (yrs)"].notna()]

    # species with both longevity and body mass data = 73 species

    overlap_df = pd.read_csv(matched_file)

    # long_species = list(matched_sp["Species"].values)
    # duplicates = [x for x in long_species if long_species.count(x) > 1]


    # 2. Check how many I have esm embeddings for
    # 152 already finished
    finished_esm_dir = '/home/gluetown/brain/data/embeddings/final_embeddings/'
    esm_emb =  [i.split("_")[0] for i in os.listdir(finished_esm_dir)]
    pre_computed = list(set(list(matched_sp["Proteome_ID"].values)) & set(esm_emb))

    # 384 species to compute
    to_download = list(set(list(matched_sp["Proteome_ID"].values)) - set(esm_emb))

    # 4. Get links to download from uniprot
    output_dir_uniprot = "/home/gluetown/viral_evo/data/longevity/uniprot/"
    get_links(to_download, output_dir_uniprot)
    ### check that they're all downloaded
    ### if unfinished:
    # get_links(list(set(to_download) - set([i.split("_")[0] for i in os.listdir(output_dir_uniprot)])), output_dir_uniprot)

    ### will need to decompress --> navigate to output_dir_uniprot and run `gzip -d *`


    # 5. Add precomputed embeddings to output dir
    input_uniprot_dir = "/home/gluetown//brain/data/embeddings/final_embeddings/"
    output_dir_esm = "/home/gluetown/viral_evo/data/longevity/esm/"
    files_to_copy = []
    for file in os.listdir(finished_esm_dir):
        if file.split("_")[0] in pre_computed:
            files_to_copy.append(input_uniprot_dir + file)
    copied, failed = copy_items_to_dir(files_to_copy, output_dir_esm)


    # 6. Create metadata file with brain + longevity data
    brain_size_file = "/home/gluetown/brain/data/metadata_all_2.csv.gz"
    brain_size_df = pd.read_csv(brain_size_file, compression="gzip")
    brain_size_df = brain_size_df.drop_duplicates(subset='Proteome_ID', keep='first')
    brain_size_df["Proteome_ID_short"] = [i.split("_")[0] for i in brain_size_df["Proteome_ID"]]
    red_longevity_df["log_body_mass"] = np.log(red_longevity_df['Body mass (g)'])
    metadata_df = pd.merge(
        red_longevity_df,
        brain_size_df[["Proteome_ID", "Proteome_ID_short", "log_body_mass", "log_brain_mass", "residuals"]], # Note the double square brackets
        left_on="Proteome_ID",
        right_on="Proteome_ID_short",
        how="left"
    )
    
    # 6.a Regress out the effect of body mass on longevity
    metadata_df['log_body_mass'] = metadata_df['log_body_mass_y'].fillna(metadata_df['log_body_mass_x'])
    metadata_df.drop(columns=['log_body_mass_y', 'log_body_mass_x'], inplace=True)
    metadata_df["log_longevity"] = np.log(metadata_df['Maximum longevity (yrs)'])
    metadata_df[metadata_df["log_body_mass"].notna() & metadata_df['Maximum longevity (yrs)'].notna()]
    residuals = regress_out(metadata_df, 'log_longevity', 'log_body_mass')
    metadata_df["longevity_residuals"] = residuals
    metadata_df.rename(columns={'Proteome_ID_x': 'Proteome_ID'}, inplace=True)
    metadata_df.drop(columns=['Proteome_ID_y'], inplace=True)

    # lonegvity_df[lonegvity_df['Body mass (g)'].notna()] # Checking


    metadata_df.to_csv("/home/gluetown/viral_evo/data/longevity/metadata.csv")
    brainsize_labels = brain_size_df[["Proteome_ID", "log_brain_mass", "residuals"]]
    brainsize_labels.to_csv("/home/gluetown/viral_evo/data/longevity/brainsize_labels.csv")
    longevity_labels = metadata_df[["Proteome_ID", 'Maximum longevity (yrs)', "log_longevity", "residuals"]]
    longevity_labels = longevity_labels[longevity_labels["residuals"].notna()]
    longevity_labels.to_csv("/home/gluetown/viral_evo/data/longevity/longevity_residuals.csv")


    # Checking overlap
    brain_size_df = pd.read_csv("/home/gluetown/viral_evo/data/longevity/brainsize_labels.csv")
    longevity_labels = pd.read_csv("/home/gluetown/viral_evo/data/longevity/longevity_labels.csv")
    longevity_residuals_df = pd.read_csv("/home/gluetown/viral_evo/data/longevity/longevity_residuals.csv")
    esm_files = os.listdir("/home/gluetown/viral_evo/data/longevity/esm/")


    # 7. Process embeddings into stack of tensors per species
    input_folder = "/home/gluetown/viral_evo/data/longevity/esm/"
    output_file = "/home/gluetown/viral_evo/data/collab/longevity/embeddings_fix.pt"
    protein_file = "/home/gluetown/viral_evo/data/collab/longevity/all_proteins_fix.json"
    process_esm_embeddings(input_folder, output_file, protein_file)


    # 8. Process protein labels

    # input_folder contains species folders with ESM embeddings
    input_folder = "/home/gluetown/viral_evo/data/longevity/esm/"
    all_proteins = {species_dir: [] for species_dir in os.listdir(input_folder)}
    for species_dir in tqdm(os.listdir(input_folder)):
        print(species_dir)
        for embedding_file in tqdm(os.listdir(os.path.join(input_folder, species_dir))):
            if embedding_file.endswith(".pt"):
                try:
                    all_proteins[species_dir].append(embedding_file.split('|')[1])
                except Exception as e:
                    print(f"Error loading embedding {embedding_file}: {e}")

    with open("/home/gluetown/viral_evo/data/longevity/all_proteins.json", "w") as f:
        json.dump(all_proteins, f)


    #### Summary of Output Files ####
    # - embeddings.pt:
    #     {species_id : torch.tensor(num_proteins, 320)}
    # - brainsize_labels.csv
    #     Proteome_ID, Maximum longevity (yrs), log_brain_mass
    # - longevity_labels.csv
    #     Proteome_ID, Maximum longevity (yrs)
    # - all_proteins.json
    #     {'Proteome_ID': ['uniprot_id1', 'uniprot_id2', ...], ...}





    ###### Debugging #####
    with open("/home/gluetown/viral_evo/data/collab/longevity/all_proteins.json", "r") as file:
        tmp = json.load(file)

    protein_names = {k.split("_")[0]:v for k,v in tmp.items()}
    len(protein_names["UP000053615"])

    embeddings = torch.load("/home/gluetown/viral_evo/data/collab/longevity/embeddings.pt")


    # check that shape of embeddings and number of genes match up
    emb_shapes = {k:v.shape[0] for k,v in embeddings.items()}
    num_proteins = {k:len(v) for k,v in protein_names.items()}

    mismatched_keys = {key: (emb_shapes[key], num_proteins[key]) for key in emb_shapes if emb_shapes[key] != num_proteins.get(key, None)}

    if mismatched_keys:
        print("Keys with mismatched values:")
        for key, (emb_shape, num_protein) in mismatched_keys.items():
            print(f"{key}: emb_shapes = {emb_shape}, num_proteins = {num_protein}")
    else:
        print("All keys have matching values.")

    """
    Keys with mismatched values:
    UP000248484: emb_shapes = 12137, num_proteins = 12138
    UP000053615: emb_shapes = 6784, num_proteins = 6785
    UP000030684: emb_shapes = 14092, num_proteins = 14093
    """

    input_folder = "/home/gluetown/viral_evo/data/longevity/esm/UP000053615_57412.fasta"

    all_embeddings = {}
    embeddings = []
    for embedding_file in tqdm(os.listdir(input_folder)):
        if embedding_file.endswith(".pt"):
            embedding = torch.load(os.path.join(input_folder, embedding_file))
            embeddings.append(embedding)
    all_embeddings[species_dir] = torch.stack(embeddings)
    # torch.save(all_embeddings, output_file)


