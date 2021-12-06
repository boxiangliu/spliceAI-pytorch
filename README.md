# Pytorch implementation of spliceAI

This is a pytorch implementation of [spliceAI](https://github.com/Illumina/SpliceAI) ([paper](https://www.cell.com/cell/pdf/S0092-8674(18)31629-5.pdf))

### Update the `data/constant.py` file
`ref_genome` should point to the hg19 genome.fa file


### Create dataset
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





