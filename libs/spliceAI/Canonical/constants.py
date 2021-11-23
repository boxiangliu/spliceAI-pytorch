CL_max=10000
# Maximum nucleotide context length (CL_max/2 on either side of the 
# position of interest)
# CL_max should be an even number

SL=5000
# Sequence length of SpliceAIs (SL+CL will be the input length and
# SL will be the output length)

splice_table='canonical_dataset.txt'
ref_genome='/genomes/Homo_sapiens/UCSC/hg19/Sequence/WholeGenomeFasta/genome.fa'
# Input details

data_dir='./'
sequence='canonical_sequence.txt'
# Output details
