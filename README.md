# NTL Thesis — Impact & Recovery Modelling

This repository contains the full pipeline for modelling the impact following typhoon events in the Philippines, using Night-Time Light (NTL) data as a proxy for impact.

---

## Project Overview

The pipeline combines satellite data (NTL radiance from satellite tiles, GHSL settlement model layers) with typhoon event records to build a predictive model of impact at grid-cell level and explore correlations between the input data and the impact data. 

The satellite data was required to be upscaled to the same level as the available ground-truth data.

A logistic regression classifier was also attempted to identify impacted cells, whose output feeds into a tree-based regression model to predict recovery metrics.

---

## Repository Structure

```
GECP_NTL_THESIS_2026/
│
├── README.md
├── pyproject.toml
├── requirements.txt
├── uv.lock
├── download_files.sh           # script to download raw data tiles from Snellius
├── get_land_cover.py           # extracts land cover layers from NC tiles and builds interpolated SMOD NetCDFs
├── run_case_study_nc.sh        # shell script to run the main NC processing pipeline
├── processing_ntl_metrics.py   # case study file to compute all of the ntl data to metrics (tifs)
│
├── data_final/                       # training data
│
├── appendix/                   # older/archived scripts and notebooks
│
├── figures/                    # figures from the analysis
│   ├── correlation analysis
│   ├── model performance
│   └── residual analysis
│
├── models/                     # pickled best models (RandomForest and LGBMRegressor and HistGradientBooster)
│
└── scripts/
    ├── helper_functions.py                 # all shared functions for data cleaning, modelling, and visualisation
    ├── logistic_regression_for_model.ipynb     # logistic regression pipeline — used standalone and as input to tree model
    ├── upscale_and_feature_engineering.ipynb  # upscales TIFs to grid cells and builds modelling dataframes
    ├── tree_model_final.ipynb                 # main model training and selection notebook
    ├── tree_model_refinements.ipynb           # refinements — land cover/urban-only baselines and oversampling experiments
    ├──correlation_analysis.ipynb              # correlation and EDA done   
    ├──quick_test_with_synthetic_data.ipynb    # process for smoothing the damage/target variable         
    ├──redisual_analysis.ipynb
    └── best_model_analysis.ipynb           # residual analysis and final model evaluation
```

---

## Requirements

Dependencies are managed with `uv`. To install:

```bash
uv sync
```

Or with pip:

```bash
pip install -r requirements.txt
```

---

## Data

Raw data is not tracked in this repository. The `download_files.sh` script handles downloading the required NTL tiles and supporting files. Processed outputs are written to `data/output_new/`.
