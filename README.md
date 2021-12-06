# Pytorch implementation of spliceAI

This is a pytorch implementation of [spliceAI](https://github.com/Illumina/SpliceAI) ([paper](https://www.cell.com/cell/pdf/S0092-8674(18)31629-5.pdf)). This repository provide the training dataset and the spliceAI model in pytorch. Please go through the following steps to reproduce the result from the spliceAI paper. 

### 0. Depdendencies
```
python=3.8.6
numpy
torch
wandb
```

### 1. Update the `data/constant.py` file
`ref_genome` should point to the hg19 genome.fa file


### 2. Create training and test dataset
```
# Get the sequence bewtween (TSS - 5000, TES + 5000) for each gene. 
bash data/grab_sequence.sh

# Create a h5 file with the following keys:
# NAME       # Gene symbol
# PARALOG    # 0 if no paralogs exist, 1 otherwise
# CHROM      # Chromosome number
# STRAND     # Strand in which the gene lies (+ or -)
# TX_START   # Position where transcription starts
# TX_END     # Position where transcription ends
# JN_START   # Positions where canonical exons end
# JN_END     # Positions where canonical exons start
# SEQ        # Nucleotide sequence
python data/create_datafile.py train all
python data/create_datafile.py test 0

python data/create_dataset.py train all 1 pytorch
python data/create_dataset.py test 0 1 pytorch
```

### 3. Train the model
```
python bin/train.py
```

### 4. Result
The model achieves 0.95 top-k accuracy and 0.98 AUPRC after 30k steps. 