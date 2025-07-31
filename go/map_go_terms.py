import requests
import os
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed

def fetch_go_terms(species, uniprot_id):
    try:
        print(f"Getting GO terms for {uniprot_id}")
        url = f"https://www.ebi.ac.uk/proteins/api/proteins/{uniprot_id}"
        response = requests.get(url, headers={"Accept": "application/json"})
        data = response.json()
        tsv_data = ""
        for feature in data['dbReferences']:
            if feature['type'] == 'GO':
                tsv_data += f"{species}\t{uniprot_id}\t{feature['id']}\t{feature['properties']['term']}\t{feature['properties']['source']}\n"
        return tsv_data
    except Exception as e:
        print(f"Error in {uniprot_id}: {e}")
        return ""

def map_go_terms(species_ids, output_dir):
    uniprot_ids = {species: [] for species in species_ids}
    uniprot_id_file = "/home/gluetown/brain/data/uniprot/"
    for species in uniprot_ids.keys():
        with open(uniprot_id_file + species, 'r') as file:
            lines = file.readlines()
        uniprot_ids[species] = [line.strip().split("|")[1] for line in lines if line.startswith(">")]

    print(f"Total number of uniprot ids: {sum([len(uniprot_ids[species]) for species in uniprot_ids.keys()])}")
    
    tot = len(species_ids)
    for i, species in enumerate(species_ids):
        tsv_data = "Species\tUniprot_id\tGO Term\tCategory\tSource\n"
        with ThreadPoolExecutor(max_workers=8) as executor:  # Use 8 threads
            futures = [executor.submit(fetch_go_terms, species, uniprot_id) for uniprot_id in uniprot_ids[species]]
            for future in as_completed(futures):
                tsv_data += future.result()
        
        print(f"Finished {species}; {i+1}/{tot}")

        with open(f"{output_dir}{species}.pkl", "wb") as w:
            pickle.dump(tsv_data, w)

if __name__ == "__main__":
    # Model Organisms
    # mouse, humans, cat, chimps, rhesus monkey, rat, horses, guinea pigs, 
    # species_ids = ["UP000000589_10090.fasta", "UP000005640_9606.fasta", "UP000002277_9598.fasta", "UP000011712_9685.fasta", "UP000006718_9544.fasta", "UP000002494_10116.fasta", 
    #                "UP000002281_9796.fasta", "UP000005447_10141.fasta"]
    # output_file = "/home/gluetown/brain/data/model.pkl"
    # map_go_terms(species_ids, output_file)

    # All Organisms
    species_ids = os.listdir("/home/gluetown/brain/data/uniprot/")
    output_dir = "/home/gluetown/brain/data/go/uniprot/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    map_go_terms(species_ids, output_dir)