#!/bin/bash

# Job settings
#SBATCH --job-name=test          
#SBATCH --account=fc_demog   
#SBATCH --partition=savio2 
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00            # Time limit (hh:mm:ss)
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=paige_park@berkeley.edu

# Load required modules
module load python/3.10.12-gcc-11.4.0 || { echo "Module load failed"; exit 1; }

# Create and activate a virtual environment
python -m venv my_env
source my_env/bin/activate

# Install all dependencies
pip install cmdstanpy arviz tensorflow pandas matplotlib numpy notebook

# Ensure cmdstan is installed
python -c "import cmdstanpy; cmdstanpy.install_cmdstan()"

# Execute the notebook
jupyter nbconvert --to notebook --execute /global/home/users/paigepark/code/bayesian_hierarchical.ipynb