# Sample Fraud Analyses

This repo explores the value of data features for identifying fraudulent purchase attempts. 

## Repo Setup

To utilize this repo, first fork and clone the repo. Then follow the proceeding instructions to finish setup on your local copy.

### Installing the conda environment and src code

The conda environment used for this repo is in the `environment.yml` file. Use the following steps to complete the installation of the environment and make code stored in src available as a package.

After cloning the repo, navigate into the repo and run:

```
# create the conda environment
conda env create -f environment.yml

# activate the conda environment
conda activate fraud-env

# make this conda environment available as a kernel in jupyter
python -m ipykernel install --user --name fraud-env --display-name "fraud-env"
```

## Guide to the Repo

The `data/` directory contains `raw/` and `processed/` subdirectories and holds all data for the project. Place a copy of the data CSV in the `data/raw/` directory to use the subsequent analytical code. 

The `notebooks/` directory holds Jupyter notebooks with Python-based code for analyzing the data. The "final" notebook is report style, and focuses on findings rather than exploratory analyses, which are found in the `exploratory/` subdirectory.

The `src/` directory holds custom Python code that can be utilized by calling the src package directly.