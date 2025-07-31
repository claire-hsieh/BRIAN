Brain mass phenotype dataset of mammals:
Protein-coding gene sequences from over 100 mammalian species were obtained from UniProt (release Release 2024_10) and processed using the ESM2 protein language model (8 million parameters) [model wrights link](https://huggingface.co/facebook/esm2_t6_8M_UR50D) to generate gene embeddings. Species-specific brain mass data were compiled from xx databases. (then talk about how did you regress out the body mass effect etc.) Each species' genome was represented as a set of gene embeddings, and bioPointNet was trained to predict relative brain mass....


- 131 species
- Not really sure when the uniprot release would be? I just used this [site](https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/reference_proteomes/Eukaryota/)?
- esm model: esm2_t6_8M_UR50D
- i think for model weights though, you just change one of the settings on the esm embedding file and it downloads it automatically
- download: https://ftp.uniprot.org/pub/databases/uniprot/knowledgebase/reference_proteomes/README 
- save as `data/uniprot_table.txt.gz` (need to gzip)

# checking num overlap
```
import pandas as pd
import torch
embeddings = torch.load("embeddings_fix.pt")
labels = pd.read_csv("brainsize_labels.csv")
len(set(labels["Proteome_ID"].to_list()).intersection(set([i.split("_")[0] for i in list(embeddings.keys())])
```

Species-specific brain mass data were compiled from the [Burger database](https://academic.oup.com/jmammal/article/100/2/276/5436908). Since brain body size allometry follows a power law, we first computed log brain and body mass values then fit them to a linear model: $brain = \alpha * (body ^ß)$, resulting in the same coefficients found in the Burger dataset ($\alpha = -1.26, \beta = 0.75$). We then used this model to calculate the residuals for each species. 

# Process
1. get species in burger dataset
2. get species in uniprot dataset
3. find overlapping species
4. regress out brain mass
	1. get log brain mass and log body mass
	2. using allometry eqn:
		1. $(Brain) = −1.26 (Body)^{0.75}$
		2. fit data to model and calculate coefficients: $brain = \alpha   * (body ^ß)$
		3. using linear model, calculate residuals for each species (data point) 
5. download uniprot proteins
6. generate ESM embeddings



## Residuals
```R
calculate_residuals <- function(df){
  model <- lm(log_brain_mass ~ log_body_mass, data = df)
  coefficients <- coef(model)
  a <- coefficients[1]
  b <- coefficients[2]
  
  # Print the coefficients
  print(paste("a:", a))
  print(paste("b:", b))
  df <- df %>%
    mutate(residuals = residuals(model))

  return(df)
}

plot_brain <- function(df){
  # Plot the data and the fitted line
  df %>%
    ggplot(aes(x = log_body_mass, y = log_brain_mass, color=source)) + 
    geom_point() + 
    labs(title = "Brain vs body mass (g)",
         x = "Body mass (log)",
         y = "Brain mass (log)") +
    geom_abline(intercept = a, slope = b, color = "red") #+ 
     #theme(legend.position = "none")
}

plot_residuals <- function(df){
  df %>%
    ggplot(aes(x = log_body_mass, y = residuals, color=source)) + 
    geom_point() +  
    labs(title = "Residuals of the log-log model",
         x = "Log Body Mass",
         y = "Residuals") #+ 
     #theme(legend.position = "none")
}
```