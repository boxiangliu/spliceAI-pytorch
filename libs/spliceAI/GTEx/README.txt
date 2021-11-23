First, update the constants.py file:
- ref_genome: path of the genome.fa file (hg19/GRCh37)

Then, use the following commands:

./grab_sequence.sh

python create_datafile.py train all
python create_datafile.py test 0

python create_dataset.py train all
python create_dataset.py test 0

qsub script_train.sh 10000 1
qsub script_train.sh 10000 2
qsub script_train.sh 10000 3
qsub script_train.sh 10000 4
qsub script_train.sh 10000 5

qsub script_test.sh 10000

# The code was tested using keras==2.0.5 and tensorflow==1.4.1
