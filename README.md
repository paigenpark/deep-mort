# deep-mort

This project investigates the efficacy of deep learning for mortality forecasting, building on the work of [Richman and Wüthrich (2021)](https://www.cambridge.org/core/journals/annals-of-actuarial-science/article/neural-network-extension-of-the-leecartermodel-to-multiple-populations/19651C62C3976DCD73C79E57CF4A071C). 

## Replication Instructions

### Data Access

The data used in this project is stored in a shared Google Drive folder. You can download it here:

👉 [Google Drive Data Folder](https://drive.google.com/drive/folders/1-ej8v9k_QCDLLW0CWtOwi5pudlqfyef0?usp=drive_link)

Clone the repo and download the data from the Google Drive link. Place the data/ folder inside the root directory of this project.

### File Structure
```
.
├── data/                        # Data folder (downloadable from Google Drive)
│
├── code/                        # Jupyter notebooks for exploration & analysis
│    ├── benchmark_models/            # Lee-Carter, Lee-Miller, and Coherent baseline implementations
│    ├── data_preparation/            # Creating clean HMD and USMDB data files from raw data & splitting data
│    ├── create_robust_figures.ipynb  # Visualization of results from paper - "robust" because uses models from multiple training runs
│    ├── evaluation_functions.py      # Functions used in create_figures.ipynb to analyze results
│    ├── train_models.ipynb           # Training deep learning models and saving model predictions
│    └── training_functions.py        # Functions used in train_models.ipynb to train models
│
├── models/                     # Saved models 
│
├── README.md                   # This file
```
### Replication Steps

If you want to replicate results from raw data: 

  1. Start with data_preparation files and create clean HMD and USMDB files.
  2. Then split the data.
  3. Train the neural networks using train_models.ipynb.
  4. Save resulting predictions from models of interest.
  5. Train benchmarks using scripts in benchmark_models directory.
  6. Replicate key figures in the working paper using create_figures.ipynb (and create your own by specifying countries of interest) 

## Citing and Contact

If you use this project in your research, please cite it as follows:
```bibtex
@misc{park2025project,
  author       = {Paige N. Park},
  title        = {Deep Learning for Mortality Forcasting},
  year         = {2025},
  howpublished = {\url{(https://github.com/paigenpark/deep-mort)}},
  note         = {Version 1.0}
}
```
If you have questions feel free to email me: paige_park@berkeley.edu
