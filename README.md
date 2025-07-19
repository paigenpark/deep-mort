# deep-mort

This project investigates the efficacy of deep learning for mortality forecasting, building on the work of [Richman and Wüthrich (2021)](https://www.cambridge.org/core/journals/annals-of-actuarial-science/article/neural-network-extension-of-the-leecartermodel-to-multiple-populations/19651C62C3976DCD73C79E57CF4A071C). 

## Replication Instructions

### Data Access

Data should be downloaded directly from the Human Mortality Database and United States Mortality DataBase according to their data sharing policies. 

To download HMD data, follow these instructions:

  1. Go to https://www.mortality.org/
  2. Register for data access if you have not previously
  3. Scroll to the bottom of the homepage and click on the Zipped Data Files button in the right corner
  4. Click on the link to the Period Death Rates zip file - this should start an automatic download
  5. Unzip the death_rates.zip file and move the data_rates folder to a data directory inside the project
  6. Rename the folder hmd_death_rates 
  
To download USMDB data, follow these instructions:
NOTE: the USMDB is transitioning to a new website at the moment. These instructions may be updated after the new webpage is available.

  1. Go to https://dataverse.harvard.edu/dataverse/usfmdb
  2. Click on the "U.S. State Life Tables" link
  3. Download the zip file
  4. Unzip the USStateLifetables2022.zip file and moved the unzipped file to the data directory in the project
  5. Rename the folder us_lifetables

### File Structure
```
.
├── data/                        # Data folder (create this and put data from HMD and USMDB inside)
│
├── code/                        # Jupyter notebooks for exploration & analysis
│    ├── benchmark_models/            # Lee-Carter, Hyndman-Ullah, and Coherent baseline implementations
│    ├── data_preparation/            # Creating clean HMD and USMDB data files from raw data & splitting data
│    ├── create_robust_figures.ipynb  # Visualization of results from paper - "robust" because uses models from multiple training runs
│    ├── evaluation_functions.py      # Functions used in create_figures.ipynb to analyze results
│    ├── train_models.ipynb           # Training deep learning models and saving model predictions
|    ├── training_functions.py        # Functions used in train_models.ipynb to train models
│    └── start_to_finish.py           # A script that runs all files to get from raw data to paper figures 
│
├── models/                      # Saved models 
│
├── renv.lock                    # Contains info about R package versions
│
├── requirements.txt             # Contains info about Python package versions
│
├── README.md                    # This file
```

### Replication Steps

If you want to replicate results from raw data: 

  1. Ensure R and Python dependencies match renv.lock and requirements.txt files (see instructions below)
  2. Run the code/start_to_finish.py file which will prepare data, train models, and reproduce paper figures
      NOTE: Training the single-country deep learning models is compute intensive so the process may take a while to run
            Using cloud/remote computing resources is recommended. 
  
### Environment Setup

R Dependencies (renv.lock)
To restore the R environment:

  1. Open R or RStudio in the project directory.

  2. Run the following:
      ```
      install.packages("renv")  # if not already installed
      renv::restore()
      ```

Python Dependencies (requirements.txt)
To set up the Python environment:

  1. Install dependencies (in a virtual environment or on your machine) by running the following bash code:
      ```
      pip install -r requirements.txt
      ```

  

## Citing and Contact

If you use this project in your research, please cite it as follows:
```bibtex
@misc{park2025project,
  author       = {Paige N. Park},
  title        = {Deep Learning for Mortality Forecasting},
  year         = {2025},
  howpublished = {\url{(https://github.com/paigenpark/deep-mort)}},
  note         = {Version 1.0}
}
```
If you have questions feel free to email me: paige_park@berkeley.edu
