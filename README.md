# Pytorch implementation of spliceAI

Under construction

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
python2 data/create_datafile.py train all
```





