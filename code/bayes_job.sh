#!/bin/bash

# Job settings
#SBATCH --job-name=test          
#SBATCH --account=fc_demog   
#SBATCH --partition=savio2 
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00            # Time limit (hh:mm:ss)

# Load required modules
module load python/3.10.12-gcc-11.4.0

jupyter nbconvert --to notebook --execute bayesian_hierarchical.ipynb
