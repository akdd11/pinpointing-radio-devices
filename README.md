# Pinpointing Radio Devices: Robust Fingerprint Localization in Industrial Environments

This repository contains the code to reproduce the results of the paper:

> Anton Schösser, Friedrich Burmeister, Joschua Bogner, Zhongju Li, Philipp Schulz, Gerhard Fettweis and Norman Franchi, "Pinpointing Radio Devices: Robust Fingerprint Localization in Industrial Environments", *submitted for publication*.

The results are based on the dataset

> F. Burmeister and A. Schösser, “Recurring, High-Precision Radio Channel Measurements in a Controlled Time-Varying Industrial Workshop Environment,” 2025. [https://dx.doi.org/10.21227/v21m-n939](https://dx.doi.org/10.21227/v21m-n939)


## Getting started

### Conda setup

First, the conda environment needs to be set up.

```bash
conda env create -f environment.yml
```

The default environment name is `pinpointing_radio_devices`.

### Dataset

Download the dataset from [https://dx.doi.org/10.21227/v21m-n939](https://dx.doi.org/10.21227/v21m-n939) and place it in the `data` folder.

### Run the localization algorithms

To run the localization algorithm, run 

```bash
python src/localization_batch.py
```

This script will run the localization for all configurations specificied in `results_to_generate.xlsx`. The configurations for the paper are already contained, further configurations can be added.

### Generate the figures

The figures for the paper are generated in the notebooks `notebooks/plot_data.ipynb` and `notebooks/plot_results.ipynb`. The figures are saved in the `figures` folder.