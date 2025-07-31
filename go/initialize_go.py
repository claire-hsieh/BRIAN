
import pandas as pd
import numpy as np
import json
import os
import pandas as pd
import torch
import time

def load_predicted_deepgo(base_dir, pattern="go"):
    if pattern == "protein":
        pattern = "protein_info"
    elif pattern == "go":
        pattern = "GO:"
    data_list = []
    deepgo_files = os.listdir(base_dir)
    for f in deepgo_files[0:8]:
        print(f)
        with open(os.path.join(base_dir, f), 'r') as file:
            json_str = ""
            for line in file:
                json_str += line.strip()
                try:
                    data = json.loads(json_str)["predictions"]
                    for item in data:
                        try: 
                            item["species"] = f.split("_deepgo")[0]
                            item["uniprot_id"] = item["protein_info"].split("|")[1]
                            data_list.append(item)
                        except:
                            continue
                    json_str = ""
                except json.JSONDecodeError:
                    continue
    go_df = pd.DataFrame(data_list)
    functions_df = go_df["functions"].apply(lambda x: pd.Series({d['name']: d['functions'] for d in x}))
    go_df = go_df.drop(columns=["functions"]).join(functions_df)
    go_df.head()
    return go_df

def tokenize_go_terms(base_dir):
    go_df = load_predicted_deepgo(base_dir, "go")
    # Tokenize GO Terms into one hot
    all_go_terms = []
    for i in ['Cellular Component', 'Molecular Function', 'Biological Process']:
        for j in go_df[i]: 
            for k in j:
                all_go_terms.append(k[0])
    all_go_terms = set()
    for i in range(go_df.shape[0]):
        for j in ['Cellular Component', 'Molecular Function', 'Biological Process']:
            all_go_terms.update([term[0] for term in go_df.iloc[i][j]])

    onehot = pd.DataFrame(0, index=go_df['uniprot_id'], columns=sorted(all_go_terms))
    for i in range(go_df.shape[0]):
        row = go_df.iloc[i]
        for j in ['Cellular Component', 'Molecular Function', 'Biological Process']:
            terms = [term[0] for term in row[j]]
            onehot.loc[row['uniprot_id'], terms] = 1
    return onehot

def format_onehot(onehot, metadata):
    grouped = metadata.groupby('species_id')
    x = {}
    protein_labels = {}
    for name, group in grouped:
        tmp_df = group[['uniprot_id']].merge(onehot, left_on='uniprot_id', right_index=True)
        if tmp_df.shape[0] != 0:
            protein_labels[name] = tmp_df['uniprot_id'].values
            x[name] = torch.Tensor(tmp_df.drop(columns=['uniprot_id']).values)
    return x, protein_labels


if __name__ == "__main__":
    time1 = time.time()
    base_dir = "/home/gluetown/brain/data/go/test_deepgo/"
    output_dir = "/home/gluetown/brain/data/go/test_go_embeddings/"
    metadata = pd.read_csv("/home/gluetown/brain/data/metadata.csv.gz", compression = "gzip")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ### LOAD DATA ###
    onehot = tokenize_go_terms(base_dir)
    x, protein_labels = format_onehot(onehot, metadata)
    onehot["uniprot_id"] = onehot.index
    onehot.index = range(onehot.shape[0])
    metadata = metadata.merge(onehot, on="uniprot_id", how="right")
    y = metadata.set_index('species_id')['Brain.resid'].dropna().to_dict()

    ### SAVE DATA ###
    torch.save(x, output_dir + "embeddings.pt")
    torch.save(y, output_dir + "labels.pt")
    torch.save(protein_labels, output_dir + "protein_labels.pt")
    time2 = time.time()
    print(f"Time taken: {time2 - time1}")