import json
import os
import pandas as pd
import torch
import time

# Load json files into dictionary

def load_predicted_deepgo(base_dir, pattern="go"):
    deepgo_files = os.listdir(base_dir)
    functions = ['Cellular Component', 'Molecular Function', 'Biological Process']
    go_terms = {f.split("_deepgo")[0]:{'Cellular Component': {"go":[], "conf":[]}, 'Molecular Function': {"go":[], "conf":[]}, 'Biological Process':{"go":[], "conf":[]}} for f in deepgo_files}
    go_labels = {func:{f.split("_deepgo")[0]:{"uniprot_id":[], 'go_terms':[]} for f in deepgo_files} for func in functions}
    all_go_terms = {} 
    for f in deepgo_files:
        species = f.split("_deepgo")[0]
        print(species)
        all_go_terms[species] = {func:[] for func in functions}
        with open(os.path.join(base_dir, f), 'r') as file:
            json_str = ""
            for line in file:
                json_str += line.strip()
                try:
                    data = json.loads(json_str)["predictions"]
                    for item in data:
                        try: 
                            for i in item["functions"]:
                                go_terms_item = [g[0] for g in i["functions"]]
                                go_labels[i["name"]][species]["uniprot_id"].append(item["protein_info"].split("|")[1])
                                go_labels[i["name"]][species]["go_terms"].append(go_terms_item)
                                go_terms[species][i["name"]]["go"].append(go_terms_item) 
                                go_terms[species][i["name"]]["conf"].append([g[2] for g in i["functions"]])
                                all_go_terms[species][i["name"]] += go_terms_item
                        except:
                            continue
                    json_str = ""
                except json.JSONDecodeError:
                    continue
            for func in functions:
                all_go_terms[species][func] = list(set(all_go_terms[species][func]))
                print(f"{func}: {len(all_go_terms[species][func])} go terms")
                print(f"{func}: {len(go_labels[func][species]["uniprot_id"])} genes ")
    return go_terms, go_labels, all_go_terms

def convert_to_binary(all_go_terms, go_terms):
    # Convert dictionary of go terms to binary matrix
    functions = ['Cellular Component', 'Molecular Function', 'Biological Process']
    final_go = {}

    # initialize empty binary matrix of same dim as all_go_terms
    for f in functions:
        final_go[f] = {}
        for species in go_terms.keys():
            final_go[f][species] = [[] for i in range(len(go_terms[species][f]["go"]))]
            for i in range(len(go_terms[species][f]["go"])):
                final_go[f][species][i] = [0 for i in range(len(all_go_terms[species][f]))]

    for species in go_terms.keys():
        for f in functions:
            for gene_index in range(len(go_terms[species][f]["go"])):
                for go_index in range(len(go_terms[species][f]["go"][gene_index])):
                    # print((species, f, gene_index, go_index))
                    go = go_terms[species][f]["go"][gene_index][go_index]
                    conf = go_terms[species][f]["conf"][gene_index][go_index]
                    final_go[f][species][gene_index][all_go_terms[species][f].index(go)] = conf
    return final_go

def load_binary_into_df(base_dir, output_dir = "", species="all", func="all"):
    # Load binary matrix into pandas dataframe
    # If specifying species, can pass in a list of species ids in format "UP_000.fasta"
    # If specifying function, can pass in a list of functions in format "cc", "mf", "bp"
    if output_dir == "":
        output_dir = base_dir

    if func == "all":
        functions = ["cc", "mf", "bp"]

    for func in ["cc", "mf", "bp"]:
        go = torch.load(f"{base_dir}/{func}_go.pt")
        labels = torch.load(f"{base_dir}/{func}_labels.pt")
        if species == "all":
            species_list = go.keys()
        elif type(species) == str:
            species_list = [species]
        for species in go.keys():
            df = pd.DataFrame(go[species], index=labels[species]["uniprot_id"], columns=labels[species]["go_terms"])
            df.to_csv(f"{output_dir}/{species}_{func}.csv")
            print(f"Finished saving {output_dir}/{species}_{func}.csv")




if __name__ == "__main__":
    ### Initialization ###
    base_dir = "/home/gluetown/brain/data/go/deepgo_0.1/"
    output_dir = "/home/gluetown/brain/data/go/deepgo_0.1_dict/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    time1 = time.time()
    go_terms, go_labels, all_go_terms = load_predicted_deepgo(base_dir, "go")
    time2 = time.time()
    print(f"Finished loading data")
    print(f"Time taken: {time2-time1}")
    time1 = time.time()
    final_go = convert_to_binary(all_go_terms, go_terms)
    print(f"Finished converting to binary")

    # Save the data
    for func in final_go.keys():
        initials = ''.join([word[0] for word in func.split()]).lower()
        torch.save(final_go[func], f"{output_dir}{initials}_go.pt")
    for func in go_labels.keys():
        initials = ''.join([word[0] for word in func.split()]).lower()
        torch.save(go_labels[func], f"{output_dir}{initials}_labels.pt")
    time2 = time.time()
    print(f"Time taken: {time2-time1}")