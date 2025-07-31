import requests
from bs4 import BeautifulSoup
import regex as re
import pandas as pd
import os

"""
def uniprot_data(species_file, uniprot_file):
    uniprot_df = pd.read_csv(uniprot_file, sep = "\t")
    uniprot_df = uniprot_df[~uniprot_df['SUPERREGNUM'].isin(['bacteria', 'viruses', 'archaea'])]
    # uniprot_df.shape
    uniprot_df['Species Name'] = uniprot_df['Species Name'].str.replace(r'\(.*?\)', '', regex=True)
    uniprot_df['Species Name'] = uniprot_df['Species Name'].str.replace(r'\d+', '', regex=True)
    uniprot_df['Species Name'] = uniprot_df['Species Name'].str.lower()

    species_name = pd.read_csv(species_file, header = None)
    species_name.columns = ['Species Name']

    prot_id = []
    for i in species_name['Species Name']:
        match = uniprot_df[uniprot_df['Species Name'].str.contains(i)]
        if len(match) > 0:
            prot_id.append(match['Proteome_ID'].values[0])
            
    base_url = "https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/reference_proteomes/Eukaryota"

    # print(len(prot_id))
    
    for i, j in zip(prot_id, uniprot_df['Tax_ID']):
        url = f"{base_url}/{i}"
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        fasta_links = [url + "/" + node.get('href') for node in soup.find_all('a') if re.search(r'\d+\.fasta\.gz$', node.get('href'))]
        for link in fasta_links:
            print(link)
            filename = link.split('/')[-1]
        if len(fasta_links) == 0:
            print(f"No fasta files found for {j}")
            # os.system(f"wget {link} -O {filename}")


if __name__ == "__main__":
    species_file = "data/brainsize_species.txt"
    # uniprot_file = "uniprot_table.txt"
    uniprot_data(species_file, sys.argv[1])
"""

def uniprot_data(species_file, uniprot_file):
    species_name = pd.read_csv(species_file, header = None)
    species_name.columns = ['Species Name']

    prot_id = []
    for i in species_name['Species Name']:
        match = uniprot_df[uniprot_df['Species Name'].str.contains(i)]
        if len(match) > 0:
            prot_id.append(match['Proteome_ID'].values[0])
    return prot_id, uniprot_df

def get_links(prot_id, output_dir ):       
    base_url = "https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/reference_proteomes/Eukaryota"

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


if __name__ == "__main__":
    species_file = "data/brainsize_species.txt"
    uniprot_file = "../data/uniprot_table.txt.gz"
    uniprot_df = pd.read_csv(uniprot_file, sep = "\t", compression='gzip')
    uniprot_df = uniprot_df[~uniprot_df['SUPERREGNUM'].isin(['bacteria', 'viruses', 'archaea'])]
    # uniprot_df.shape
    uniprot_df['Species Name'] = uniprot_df['Species Name'].str.replace(r'\(.*?\)', '', regex=True)
    uniprot_df['Species Name'] = uniprot_df['Species Name'].str.replace(r'\d+', '', regex=True)
    uniprot_df['Species Name'] = uniprot_df['Species Name'].str.lower()

    # prot_id, uniprot_df = uniprot_data(species_file, sys.argv[1])
    ids = ["UP000504623","UP000009136","UP000694520","UP000030684","UP000805418","UP000252040","UP000248484","UP000189704","UP000233040","UP000504640","UP000005215","UP000694417","UP000001075"]
    get_links(ids, uniprot_df)

