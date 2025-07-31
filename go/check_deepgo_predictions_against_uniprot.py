import pandas as pd
from io import StringIO
import json
import os
import pandas as pd

# human: UP000005640_9606

def load_uniprot_go_terms(uniprot_dir, species_id):
    uniprot = pd.read_pickle(f"{uniprot_dir}{species_id}.pkl")
    data = StringIO(uniprot)
    uniprot_df = pd.read_csv(data, sep="\t")
    print(f"Number of GO terms: {uniprot_df.shape[0]}")
    uniprot_df["Source"] = uniprot_df["Source"].apply(lambda x: x.split(":")[0])
    experimentally_derived_sources = ["EXP", "IDA", "IPI", "IMP", "IGI", "IEP"]
    # exp_derived_srcs_deepgo = ["EXP","IDA","IPI","IMP","IGI","IEP","TAS","IC","HTP","HDA","HMP","HGI","HEP"]
    uniprot_df = uniprot_df.loc[uniprot_df["Source"].isin(experimentally_derived_sources)]
    print(f"Number of GO terms after filtering for experimental sources: {uniprot_df.shape[0]}")  # 103566 after filtering source
    correct = uniprot_df.groupby('Uniprot_id')['GO Term'].apply(list).to_dict()
    return correct


def load_deepgo_predicted(base_dir, species_id):
    data_list = []
    deepgo_files = [f"{species_id}_deepgo.json"]
    

    for f in deepgo_files:

        with open(base_dir + f, 'r') as file:
            data = json.load(file)["predictions"]
            for item in data:
                item['file'] = f.split("_deepgo")[0]  # Add a column for the file identifier
                data_list.append(item)
    go_df = pd.DataFrame(data_list)
    functions_df = pd.json_normalize(go_df['functions'])



    go_df = go_df.drop(columns=['functions']).join(functions_df)
    go_df.rename(columns={0: "Cellular Component", 1: 'Molecular Function', 2:  'Biological Process',}, inplace=True)
    for i in ['Cellular Component', 'Molecular Function', 'Biological Process']:
        go_df[i] = go_df[i].apply(lambda x: x['functions'] if isinstance(x, dict) and 'functions' in x else x)
    go_df["uniprot_id"] = go_df["protein_info"].apply(lambda x: x.split("|")[1] if isinstance(x, str) else x)
    go_df.drop(["protein_info", "sequence", "file"], axis=1, inplace=True)
    go_df['all_go'] = go_df.apply(extract_first_gene, axis=1)

    predicted = go_df.groupby('uniprot_id')['all_go'].apply(list).to_dict()
    return predicted


# Define the function to extract the first element of each gene
def extract_first_gene(row):
    all_go_terms = []
    for func in ['Cellular Component', 'Molecular Function', 'Biological Process']:
        if isinstance(row[func], list):
            all_go_terms += [gene[0] for gene in row[func] if isinstance(gene, list) and len(gene) > 0]
    return all_go_terms

def check_accuracy(correct, predicted):
    total_go_terms = sum(len(sublist) for sublist in list(correct.values()))

    correct_predictions = 0
    for uniprot_id, correct_go_term in correct.items():
        for go_term in correct_go_term:
            try:
                if go_term in predicted[uniprot_id]:
                    correct_predictions += 1
            except:
                continue
    print(f"Accuracy: {correct_predictions/total_go_terms}")


if __name__ == "__main__":
    base_dir = "/home/gluetown/brain/data/go/deepgo_0.1/"
    uniprot_dir = "/home/gluetown/brain/data/uniprot/"
    species_id = "UP000005640_9606.fasta"
    correct = load_uniprot_go_terms(uniprot_dir, species_id)
    predicted = load_deepgo_predicted(base_dir, species_id)
    check_accuracy(correct, predicted)
