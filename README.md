# NTL Thesis — Impact & Recovery Modelling

This repository contains the full pipeline for modelling the impact and recovery following typhoon events in the Philippines, using Night-Time Light (NTL) data as a proxy for impact.

---

## Project Overview

The pipeline combines remote sensing data (NTL radiance from satellite tiles, GHSL settlement model layers) with typhoon event records to build a predictive model of recovery duration at grid-cell level. A logistic regression classifier is used to identify impacted cells, whose output feeds into a tree-based regression model to predict recovery metrics.

---

## Repository Structure

```
ntlthesis/
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
├── data/                       # raw and processed data (not tracked in git)
│
├── appendix/                   # older/archived scripts and notebooks
│
├── column_selections/          # CSV files of selected feature columns for each model variant
│
├── figures/                    # generated figures
│   ├── correlation analysis
│   ├── model performance
│   └── residual analysis
│
├── models/                     # pickled best models (RandomForest and LGBMRegressor
│
└── scripts/
    ├── helper_functions.py                 # all shared functions for data cleaning, modelling, and visualisation
    ├── logistic_regression_model.ipynb     # logistic regression pipeline — used standalone and as input to tree model
    ├── upscale_and_feature_engineering.ipynb  # upscales TIFs to grid cells and builds modelling dataframes
    ├── tree_model_final.ipynb              # main model training and selection notebook
    ├── tree_model_refinements.ipynb        # refinements — land cover/urban-only baselines and oversampling experiments
    └── best_model_analysis.ipynb           # residual analysis and final model evaluation
```

---

## Pipeline

The pipeline runs in the following order:

**1. Data extraction** — `get_land_cover.py` and `run_case_study_nc.sh` (`processing_ntl_metrics.py`) extract NTL and land cover data from raw NC tiles, apply spatial corrections, and save outputs as GeoTIFFs. GHSL SMOD layers are extracted and linearly interpolated across years to produce storm-year-specific urban/suburban masks.

**2. Feature engineering** — `upscale_and_feature_engineering.ipynb` reads the GeoTIFFs, performs zonal statistics against a grid vector, and assembles per-storm per-cell dataframes. Two variants are produced: a full dataset and an urban/suburban-masked dataset.

**3. Logistic regression** — `logistic_regression_model.ipynb` trains a logistic regression classifier to predict whether a grid cell was impacted. The output probabilities are used as an additional feature in the tree model.

**4. Tree model training** — `tree_model_final.ipynb` trains and evaluates multiple ensemble regressors (Random Forest, Extra Trees, LGBM, HGB, XGBoost) to predict recovery duration. Includes oversampling of tail values, feature selection, scaling, log-transform of the target, and gridsearch.

**5. Refinements** — `tree_model_refinements.ipynb` runs additional experiments including land cover and urban-only dataset comparisons, further oversampling configurations, and final model selection.

**6. Analysis** — `best_model_analysis.ipynb` performs residual analysis, evaluates the best model, and generates figures for reporting.

---

## Key Design Decisions

- **Two-stage modelling**: logistic regression first identifies impacted cells; tree regression then predicts recovery duration only where the signal is meaningful.
- **Urban/suburban masking**: GHSL SMOD layers are interpolated to the storm year and used to mask zonal statistics, focusing the model on settled areas.
- **Oversampling**: tail values of the target distribution are augmented with small Gaussian noise to improve model performance on extreme recovery durations.
- **Feature selection**: correlated feature groups are identified via graph-based clustering and the most informative representative from each group is selected.
- **Gap-aware model selection**: improvements are only accepted if the CV–test R² gap stays within a configurable threshold, preventing overfitting from being rewarded.

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
