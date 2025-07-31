import requests
import json
import os
import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

def fetch_predictions(url, headers, payload, output_file):
    try:
        response = requests.post(url, headers=headers, json=payload)
        with open(output_file, 'a') as f:
            json.dump(response.json(), f, indent=4)
    except Exception as e:
        print(f"Error occurred: {e}")
        print(f"Failed to get predictions for {output_file}")
        print(f"Payload: {payload}")
        print(f"Response: {response}")

def get_deepgo_predictions(species_ids, output_dir, threshold=0.3):
    url = "https://deepgo.cbrc.kaust.edu.sa/deepgo/api/create"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    headers = {
        "Content-Type": "application/json",
    }
    uniprot_filepath = "/home/gluetown/brain/data/uniprot/"
    for species in species_ids: 
        print(f"Getting deepgo predictions for {species}")
        with open(uniprot_filepath + species, 'r') as file:
            lines = file.readlines()
            output_file = os.path.join(output_dir, f"{species}_deepgo.json")
            with ThreadPoolExecutor(max_workers=8) as executor:  # Use 8 threads
                futures = []
                for i in tqdm.tqdm(range(0, len(lines), 100)):
                    chunk = lines[i:i+100]
                    fasta_input = "".join(chunk)
                    payload = {
                        "version": "1.0.20",
                        "data_format": "fasta",
                        "data": fasta_input,
                        "threshold": threshold
                    }
                    futures.append(executor.submit(fetch_predictions, url, headers, payload, output_file))
                for future in as_completed(futures):
                    future.result()

if __name__ == "__main__":
    threshold = 0.1
    output_dir = f"/home/gluetown/brain/data/go/deepgo_{threshold}/"
    species_ids = [i for i in os.listdir("/home/gluetown/brain/data/uniprot/") if i not in os.listdir(f"/home/gluetown/brain/data/go/model_deepgo_predicted_{threshold}/")]
    species_ids = [i for i in species_ids if i not in os.listdir(output_dir)]
    get_deepgo_predictions(species_ids, output_dir, threshold=threshold)

