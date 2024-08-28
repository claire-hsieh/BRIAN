# BRIAN

BRAIN: Bagging for Recognition and Identification of Associations with Neural Size

## Data: 
- ESM2 embeddings created from uniprot protein sequences of vertebrate species belonging to the brain size dataset from this [paper](https://academic.oup.com/jmammal/article/100/2/276/5436908)

## Goal
- tbd

## Organization
- `scripts/get_esm.py`  
  
  creates ESM2 embeddings using the [ESM](https://github.com/facebookresearch/esm) model  
  called by `scripts/get_esm.sh` but you need to first download the appropriate proteins from [uniprot](https://ftp.uniprot.org/pub/databases/uniprot/knowledgebase/reference_proteomes/) and filter out the species that are in the database using ` python3 uniprot_data.py > link.txt`
  

  
  
  