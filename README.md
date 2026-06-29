# deep-mort

This project investigates why and in what contexts a simple, out-of-the-box MLP deep learning model outperforms the tailored time series approaches foundational to the field of demography. Read [my preprint](https://osf.io/preprints/socarxiv/sqphx_v2) to find out! 

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
├── code/                        # Notebooks and scripts for data prep, training & analysis
│    ├── data_preparation/                  # Create clean HMD/USMDB data files from raw data & split data
│    │    ├── create-hmd-file.R
│    │    ├── create-usmdb-file.R
│    │    └── split_data.py
│    ├── benchmark_models/                  # Lee-Carter, Hyndman-Ullah, and Coherent baseline implementations (R)
│    ├── train_dl_models.ipynb              # Train deep learning models and save model predictions
│    ├── training_functions.py              # Functions used in train_dl_models.ipynb
│    ├── create_figures_tables_1-3.ipynb    # Main paper figures and tables
│    ├── evaluation_functions.py            # Functions used in the figure notebooks to analyze results
│    ├── uncertainty_models/                # Uncertainty-quantification model training & evaluation
│    │    ├── train_dl_models_freeze_uncertainty.ipynb
│    │    ├── training_functions_freeze_uncertainty.py
│    │    └── evaluation_uncertainty_fig_4.ipynb
│    ├── supplemental_figures/              # Supplementary 100-year forecasts and figures
│         ├── lee-carter_100.r, hyndman-ullah_100.R, coherent_100.r
│         ├── supplement_figures.ipynb
│         ├── code_expanding_window/             # Expanding-window pipeline used for the supplement
│         └── supp_data/                         # Saved 100-year forecasts from each model
│              
│
├── figures/                     # Generated paper figures (PNG)
│
├── models/                      # Trained models (.keras) are written here by the training notebooks (not tracked in git)
│
├── renv.lock                    # Contains info about R package versions
│
├── requirements.txt             # Contains info about Python package versions
│
├── README.md                    # This file
```

### Replication Steps

If you want to replicate results from raw data, first ensure your R and Python dependencies match the `renv.lock` and `requirements.txt` files (see [Environment Setup](#environment-setup) below). Then run the scripts in the order listed.

> **NOTE:** Training the single-country deep learning models is compute intensive and may take a while to run. Using cloud/remote computing resources is recommended.

A few conventions:
  - The R scripts use the [`here`](https://here.r-lib.org/) package, so they can be run from anywhere in the project.
  - The Jupyter notebooks use relative paths and should be run with their own containing directory as the working directory (e.g. run `train_dl_models.ipynb` from `code/`, and run the supplemental notebooks from their respective subfolders).

#### Main results (Figures 1–3 and Tables)

  1. **Build clean data files from the raw HMD/USMDB downloads:**
      ```bash
      Rscript code/data_preparation/create-hmd-file.R
      Rscript code/data_preparation/create-usmdb-file.R
      ```
  2. **Create the train/validation/test splits:**
      ```bash
      python code/data_preparation/split_data.py
      ```
  3. **Run the benchmark time-series models:**
      ```bash
      Rscript code/benchmark_models/lee-carter.r
      Rscript code/benchmark_models/hyndman-ullah.R
      Rscript code/benchmark_models/coherent.r
      ```
  4. **Train the deep learning models and save their predictions:** run `code/train_dl_models.ipynb`.
  5. **Generate the main figures and tables:** run `code/create_figures_tables_1-3.ipynb`. Figures are written to `figures/`.

#### Uncertainty analysis (Figure 4)

Builds on the data and benchmark forecasts produced in the main pipeline above.

  6. Train the uncertainty-quantification models: run `code/uncertainty_models/train_dl_models_freeze_uncertainty.ipynb`.
  7. Produce Figure 4: run `code/uncertainty_models/evaluation_uncertainty_fig_4.ipynb`.

#### Supplemental figures (100-year forecasts and expanding-window analysis)

  8. Regenerate the 100-year benchmark forecasts:
      ```bash
      Rscript code/supplemental_figures/lee-carter_100.r
      Rscript code/supplemental_figures/hyndman-ullah_100.R
      Rscript code/supplemental_figures/coherent_100.r
      ```
  9. Produce the supplemental figures: run `code/supplemental_figures/supplement_figures.ipynb` (reads the saved forecasts in `code/supplemental_figures/supp_data/`).

The expanding-window pipeline used in the supplement lives in `code/supplemental_figures/code_expanding_window/` and mirrors the main pipeline above.

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

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Citing and Contact

If you use this project in your research, please cite it as follows:
```bibtex
@misc{park2026project,
  author       = {Paige N. Park},
  title        = {Deep Learning for Mortality Forecasting},
  year         = {2026},
  howpublished = {\url{https://github.com/paigenpark/deep-mort}},
  note         = {Version 1.0}
}
```
If you have questions feel free to email me: paige_park@berkeley.edu
