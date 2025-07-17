import subprocess
import os
# Set working directory to script's directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# create single hmd file with data from all countries
subprocess.run(["Rscript", "data_preparation/create-hmd-file.R"])

# create single usmdb file with data from all US states
subprocess.run(["Rscript", "data_preparation/create-usmdb-file.R"])

# split hmd and usmdb into training and test sets
subprocess.run(["python", "data_preparation/split_data.py"])

# run 5 iterations of each benchmark model using hmd data and save predictions
subprocess.run(["Rscript", "benchmark_models/lee-carter.R"])
subprocess.run(["Rscript", "benchmark_models/hyndman-ullah.R"])
subprocess.run(["Rscript", "benchmark_models/coherent.R"])

# train deep learning models using hmd data and save predictions
subprocess.run(["jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace", 
                "train_dl_models.ipynb"], check=True)

# generate paper figures using predictions from benchmark and deep learning models
subprocess.run(["jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace", 
                "create_robust_figures.ipynb"], check=True)

