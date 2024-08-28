import os
import concurrent.futures
from pathlib import Path
import torch
import time
import sys

from esm import FastaBatchedDataset, pretrained, MSATransformer

"""
From ESM code:
Run commands in parallel: 
python scripts/extract.py esm2_t33_650M_UR50D examples/data/some_proteins.fasta examples/data/some_proteins_emb_esm2 --repr_layers 0 32 33 --include mean
"""

# Adapted from ESM scripts/extract.py
def run(model_location, fasta_file, output_dir):
    model, alphabet = pretrained.load_model_and_alphabet(model_location)
    model.eval()

    if isinstance(model, MSATransformer):
        raise ValueError(
            "This script currently does not handle models with MSA input (MSA Transformer)."
        )
    
    if torch.cuda.is_available():
        model = model.cuda()
        print("Transferred model to GPU")

    dataset = FastaBatchedDataset.from_file(fasta_file)
    batches = dataset.get_batch_indices(4096, extra_toks_per_seq=1)
    data_loader = torch.utils.data.DataLoader(
        dataset, collate_fn=alphabet.get_batch_converter(1022), batch_sampler=batches
    )
    print(f"Read {fasta_file} with {len(dataset)} sequences")
    
    output_dir.mkdir(parents=True, exist_ok=True)

    assert all(-(model.num_layers + 1) <= i <= model.num_layers for i in [0, 5, 6])
    repr_layers = [(i + model.num_layers + 1) % (model.num_layers + 1) for i in [0, 5, 6]]

    with torch.no_grad():
        for batch_idx, (labels, strs, toks) in enumerate(data_loader):
            # print(
            #     f"Processing {batch_idx + 1} of {len(batches)} batches ({toks.size(0)} sequences)"
            # )
            if torch.cuda.is_available():
                toks = toks.to(device="cuda", non_blocking=True)

            out = model(toks, repr_layers=repr_layers, return_contacts=False)

            logits = out["logits"].to(device="cpu")
            representations = {
                layer: t.to(device="cpu") for layer, t in out["representations"].items()
            }

            for i, label in enumerate(labels):
                output_file = output_dir / f"{label}.pt"
                # print(output_file)
                output_file.parent.mkdir(parents=True, exist_ok=True)
                result = {"label": label}
                truncate_len = min(1022, len(strs[i]))
                # Call clone on tensors to ensure tensors are not views into a larger representation
                # See https://github.com/pytorch/pytorch/issues/1995

                result["mean_representations"] = {
                    layer: t[i, 1 : truncate_len + 1].mean(0).clone()
                    for layer, t in representations.items()
                }

                torch.save(
                    result,
                    output_file,
                )
	# print(f"Finished {fasta_file}")

def generate_embedding(file_download): 
    species_name = file_download 
    file_directory = f"/home/gluetown/brain/uniprot/" 
    file_directory_output = f"./uniprot_embeddings4/"
    file_path = f"{file_directory}/{file_download}"
    file_out_path = f"{file_directory_output}/{species_name}"
    run("esm2_t6_8M_UR50D", Path(file_path), Path(file_out_path))

def download_all_files(num_species, max_workers): 
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i in range(num_species):
            futures.append(executor.submit(generate_embedding, file_names_fa[i]))
            time.sleep(1)

    concurrent.futures.wait(futures)

if __name__ == "__main__":
    num_species = 10
    max_workers = 10
    file_names_fa = sorted([os.listdir("/home/gluetown/brain/uniprot/")][0])[int(sys.argv[1]):int(sys.argv[2])]
    # file_names_fa = sorted([os.listdir("/home/gluetown/brain/uniprot/")][0])
    print(file_names_fa)
    start_time = time.time()
    download_all_files(num_species, max_workers)
    for file in file_names_fa:
        generate_embedding(file)
    print("--- %s seconds ---" % (time.time() - start_time))

