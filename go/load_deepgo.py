import os
import json
import pandas as pd

def load_predicted_deepgo(base_dir, pattern="go"):
    if pattern == "protein":
        pattern = "protein_info"
    elif pattern == "go":
        pattern = "GO:"
    data_list = []
    deepgo_files = os.listdir(base_dir)
    for f in deepgo_files:
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

if __name__ == "__main__":
    base_dir = "/home/gluetown/brain/data/go/model_deepgo_predicted_0.1/"
    go_df = load_predicted_deepgo(base_dir)
    go_df.to_csv(base_dir + "csv.gz", index=False, compression="gzip")