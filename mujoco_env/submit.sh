#!/bin/bash

#SBATCH --ntasks 1
#SBATCH --cpus-per-task 16
#SBATCH --mem=15000
#SBATCH -o outfile  # send stdout to outfile
#SBATCH -e errfile  # send stderr to errfile
#SBATCH -t 1500:00:00

./staskfarm ${1}


