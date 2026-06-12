import subprocess
import os
from datetime import datetime

# Set working directory to script's directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Create a log file in the logs directory with a timestamp
log_path = os.path.join("logs", f"process_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
log_file = open(log_path, "w")

print(f"Logging to: {log_path}")

# Run process function
def run_process(command, description=None):
    log_file.write(f"\nRunning: {' '.join(command)}\n")
    if description:
        log_file.write(f"Description: {description}\n")
    try:
        subprocess.run(command, stdout=log_file, stderr=log_file, check=True)
        log_file.write(f"Completed: {' '.join(command)}\n")
    except subprocess.CalledProcessError as e:
        log_file.write(f"ERROR during: {' '.join(command)}\n")
        log_file.write(f"Return code: {e.returncode}\n")
    log_file.flush() 


# Run each step with logging
run_process(["Rscript", "data_preparation/create-hmd-file.R"], "Create HMD file")
run_process(["Rscript", "data_preparation/create-usmdb-file.R"], "Create USMDB file")
run_process(["python", "data_preparation/split_data.py"], "Split data")
run_process(["Rscript", "benchmark_models/lee-carter.R"], "Run Lee-Carter model")
run_process(["Rscript", "benchmark_models/hyndman-ullah.R"], "Run Hyndman-Ullah model")
run_process(["Rscript", "benchmark_models/coherent.R"], "Run Coherent model")
run_process(["jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace", 
             "train_dl_models.ipynb"], "Train deep learning models")
run_process(["jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace", 
             "create_robust_figures.ipynb"], "Generate figures")

log_file.close()

