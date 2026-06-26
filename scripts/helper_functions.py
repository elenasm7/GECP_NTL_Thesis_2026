# standard library
import copy
import csv
import json
import os
import pickle
import re
import sys
import shutil
import warnings
from datetime import datetime, timedelta
from itertools import product
from pathlib import Path
from typing import List

# data
import numpy as np
import pandas as pd
import xarray as xr

# geospatial
import geopandas as gpd
import rasterio
from rasterio.merge import merge
import rioxarray
import shapefile
from osgeo import ogr
from rasterio.crs import CRS
from rasterstats import zonal_stats
from shapely import wkt
from shapely.geometry import Point, box, shape
from shapely.strtree import STRtree

# viz
import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
import matplotlib.cm as cm

# ML
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import (
    BaggingRegressor, ExtraTreesRegressor, GradientBoostingRegressor,
    HistGradientBoostingRegressor, RandomForestRegressor,
)
from sklearn.exceptions import DataConversionWarning
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, root_mean_squared_error
from sklearn.model_selection import (
    GridSearchCV, RandomizedSearchCV, RepeatedKFold,
    cross_val_score, train_test_split,
)
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVR

# misc
import networkx as nx
import psutil
import tempfile

# config
matplotlib.use('Agg')
plt.rcParams['font.family'] = 'Helvetica'
ogr.UseExceptions()
warnings.filterwarnings('ignore', category=DataConversionWarning)

BACKGROUND_COLOR = '#EAF2EF'

thesis_cmap_div = mcolors.LinearSegmentedColormap.from_list(
    "thesis_div",
    [
        '#F6511D',  # orange red
        '#EAF2EF',  # off white
        '#097FC7',  # ocean blue
    ]
)


# upscaling

def upscale_data(gdf, input_raster, method=["mean"], duration=None):
    '''
    Takes a grid geodataframe and a raster file, calculates the specified zonal
    statistic for each grid cell, and returns the results as a list. 
    method can be "mean", "median", "std", "categorical", or "percentage_avail".
    '''
    with rasterio.open(input_raster) as src:
        data = src.read(1).astype(float)
        for nodata_val in [-9999, 888, 999, 9999]:
            data[data == nodata_val] = np.nan
        profile = src.profile.copy()
        profile.update({"dtype": "float64", "nodata": np.nan})
        temp_file = "../data/temp_masked.tif"
        with rasterio.open(temp_file, "w", **profile) as dst:
            dst.write(data, 1)

    if method == 'categorical':
        stats = zonal_stats(gdf, temp_file, stats=[], categorical=True, nodata=np.nan)
    elif method == 'percentage_avail':
        sum_stats = zonal_stats(gdf, temp_file, stats=["sum", "count"], nodata=np.nan)
        stats = [(s["sum"] / (s["count"] * duration)) if s["count"] and s["count"] > 0 else None for s in sum_stats]
    else:
        stats_org = zonal_stats(gdf, temp_file, stats=method, nodata=np.nan)
        stats = [s[method] if s[method] is not None else None for s in stats_org]

    return stats


def flag_proportion(gdf, input_raster, flag_value):
    '''
    Returns the proportion of pixels in each cell equal to flag_value,
    as a decimal between 0 and 1.
    '''
    with rasterio.open(input_raster) as src:
        data = src.read(1).astype(float)
        if flag_value == -9999:
            flag_raster = np.where((data == flag_value) | (np.isnan(data)), 1.0, 0.0)
        else:
            flag_raster = np.where(data == flag_value, 1.0, 0.0)
        profile = src.profile.copy()
        profile.update({"dtype": "float64"})
        temp_file = "../data/temp_flag.tif"
        with rasterio.open(temp_file, "w", **profile) as dst:
            dst.write(flag_raster, 1)
    stats = zonal_stats(gdf, temp_file, stats=["median"], nodata=np.nan)
    return [s["median"] for s in stats]


def affected_perc_and_bool(aff_col):
    '''
    Takes the affected categories column (format: "{0: 0.0, 1: 0.5, 2: 0.5}") 
    and returns the percentage of affected cells and boolean flags as a tuple.
    '''
    if aff_col is None or aff_col == '' or aff_col == '{}':
        return 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    cat_dict = {i.split(':')[0].strip(): i.split(':')[1].strip() for i in aff_col.replace('{', '').replace('}', '').split(',')}
    total_cells = sum(float(v) for v in cat_dict.values())
    affected_cells = sum(float(v) for k, v in cat_dict.items() if k in ['1.0', '2.0'])
    possibly_aff_cells = sum(float(v) for k, v in cat_dict.items() if k in ['2.0'])
    cert_aff_cells = sum(float(v) for k, v in cat_dict.items() if k in ['1.0'])
    not_affected_cells = sum(float(v) for k, v in cat_dict.items() if k in ['3.0'])
    four_affected_cells = sum(float(v) for k, v in cat_dict.items() if k in ['4.0'])
    zero_affected_cells = sum(float(v) for k, v in cat_dict.items() if k in ['0.0'])
    affected_perc = (affected_cells / total_cells) if total_cells > 0 else 0
    cert_not_affected_perc = (not_affected_cells / total_cells) if total_cells > 0 else 0
    poss_affected_perc = (possibly_aff_cells / total_cells) if total_cells > 0 else 0
    cert_aff_perc = (cert_aff_cells / total_cells) if total_cells > 0 else 0
    four_aff_perc = (four_affected_cells / total_cells) if total_cells > 0 else 0
    zero_aff_perc = (zero_affected_cells / total_cells) if total_cells > 0 else 0
    cert_aff = 1 if cert_aff_cells > 0 else 0
    prob_aff = 1 if affected_cells > 0 else 0
    cert_not_aff = 1 if not_affected_cells > 0 else 0
    return 0, affected_perc, cert_aff, prob_aff, cert_not_aff, cert_not_affected_perc, poss_affected_perc, cert_aff_perc, four_aff_perc, zero_aff_perc, total_cells


def write_temp_tif(data_2d, haz_tif, nodata=np.nan):
    if isinstance(haz_tif, str):
        with rasterio.open(haz_tif) as src:
            profile = src.profile.copy()
    else:
        profile = haz_tif.profile.copy()
    profile.update({'dtype': 'float64', 'nodata': nodata, 'count': 1})
    tmp = tempfile.NamedTemporaryFile(suffix='.tif', delete=False)
    with rasterio.open(tmp.name, 'w', **profile) as dst:
        dst.write(data_2d, 1)
    return tmp.name


def mask_tif_with_urban(input_tif_path, urban_mask_array):
    with rasterio.open(input_tif_path) as src:
        data = src.read(1).astype(float)
        profile = src.profile.copy()
    data[urban_mask_array != 1] = np.nan
    data[np.isnan(urban_mask_array)] = np.nan
    profile.update({'dtype': 'float64', 'nodata': np.nan})
    tmp = tempfile.NamedTemporaryFile(suffix='.tif', delete=False)
    with rasterio.open(tmp.name, 'w', **profile) as dst:
        dst.write(data, 1)
    return tmp.name


# null / missingness helpers

def find_null_values_in_cols(df):
    nas = df.isna().sum()
    return nas[nas > 0]


def get_na_columns(df):
    nas = df.isna().sum()
    return nas[nas > 0].keys()


def find_bad_storms(df, na_columns):
    # find storms where ALL values are null for any of the na cols
    bad_storms = []
    for storm in df['typhoon_name'].unique():
        for col in na_columns:
            if df[df['typhoon_name'] == storm][col].isna().all():
                bad_storms.append(storm)
                break
    return list(set(bad_storms))


def describe_row_missingness(df):
    '''
    Prints a summary of missing values per row and plots the distribution.
    '''
    n_cols = df.shape[1]
    row_miss = df.isna().sum(axis=1) / n_cols

    print("Row missingness distribution: ")
    print(row_miss.describe().round(4))
    print(f"\nRows with any missing:  {(row_miss > 0).sum()} ({(row_miss > 0).mean()*100:.1f}%)")
    print(f"Rows with >10% missing: {(row_miss > 0.10).sum()} ({(row_miss > 0.10).mean()*100:.1f}%)")
    print(f"Rows with >25% missing: {(row_miss > 0.25).sum()} ({(row_miss > 0.25).mean()*100:.1f}%)")
    print(f"Rows with >50% missing: {(row_miss > 0.50).sum()} ({(row_miss > 0.50).mean()*100:.1f}%)")
    print(f"Rows with 100% missing: {(row_miss == 1.0).sum()} ({(row_miss == 1.0).mean()*100:.1f}%)")

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor(BACKGROUND_COLOR)
    ax.set_facecolor(BACKGROUND_COLOR)
    ax.hist(row_miss, bins=50, color='#097FC7', edgecolor='none')
    ax.axvline(row_miss.mean(), color='#F6511D', linestyle='--', lw=1.5, label=f'Mean: {row_miss.mean():.2f}')
    ax.set_xlabel("Proportion of missing values per row")
    ax.set_ylabel("Count")
    ax.set_title("Row-level missingness distribution")
    ax.legend()
    plt.tight_layout()
    plt.savefig('../figures/row_missingness.png', dpi=150, bbox_inches='tight', facecolor=BACKGROUND_COLOR)
    plt.show()
    return row_miss


def drop_rows_by_missingness(df, threshold):
    '''
    Drops rows where missingness proportion exceeds threshold (0-1).
    '''
    row_miss = df.isna().sum(axis=1) / df.shape[1]
    mask = row_miss <= threshold
    df_clean = df[mask]
    print(f"Dropped {(~mask).sum()} rows above {threshold*100:.0f}% missingness threshold")
    print(f"Remaining: {len(df_clean)} from {len(df)}")
    return df_clean


# Null imputation

def calculate_storm_medians(df, cols):
    '''
    Calculates storm-specific medians split by affected/not-affected status,
    returns a nested dict for use in imputation.
    '''
    storm_medians = {}
    for storm in df['typhoon_name'].unique():
        storm_medians[storm] = {}
        for col in cols:
            storm_medians[storm][col] = {
                'prob_aff':      df[(df['typhoon_name'] == storm) & (df['prob_aff'] == 1)][col].median(),
                'not_affected':  df[(df['typhoon_name'] == storm) & (df['prob_aff'] == 0)][col].median(),
                'overall_median': df[df['typhoon_name'] == storm][col].median(),
            }
    return storm_medians


def simp_calculate_storm_medians(df, cols):
    '''
    Calculates the overall storm median for each column, no affected split.
    '''
    storm_medians = {}
    for storm in df['typhoon_name'].unique():
        storm_medians[storm] = {}
        for col in cols:
            storm_medians[storm][col] = df[df['typhoon_name'] == storm][col].median()
    return storm_medians


def get_storm_median_for_row(row, storm_medians, col):
    '''
    Returns the appropriate storm-specific median for a row based on prob_aff status.
    Falls back to overall median if the affected-specific median is null.
    '''
    storm_name = row['typhoon_name']
    if storm_name not in storm_medians:
        return np.nan
    medians = storm_medians[storm_name][col]
    if row['prob_aff'] == 1 and not pd.isna(medians['prob_aff']):
        return medians['prob_aff']
    elif row['prob_aff'] == 0 and not pd.isna(medians['not_affected']):
        return medians['not_affected']
    else:
        return medians['overall_median']


def get_storm_median_for_row_more_complex(row, storm_medians, col):
    '''
    Imputes using pot_affected_perc to interpolate between the not-affected
    and affected medians, giving a more granular value than the binary prob_aff split.
    '''
    storm_name = row['typhoon_name']
    if storm_name not in storm_medians:
        return np.nan
    medians = storm_medians[storm_name][col]
    pot_aff = row.get('pot_affected_perc', np.nan)
    if not pd.isna(pot_aff) and not pd.isna(medians['prob_aff']) and not pd.isna(medians['not_affected']):
        return medians['not_affected'] + pot_aff * (medians['prob_aff'] - medians['not_affected'])
    if row['prob_aff'] == 1 and not pd.isna(medians['prob_aff']):
        return medians['prob_aff']
    elif row['prob_aff'] == 0 and not pd.isna(medians['not_affected']):
        return medians['not_affected']
    else:
        return medians['overall_median']


def impute_with_storm_medians(df, storm_medians, cols, global_medians=None):
    '''
    Replaces null values in cols with storm-specific medians.
    Falls back to global median for storms not seen during training.
    '''
    df = df.copy()
    for col in cols:
        for storm in df['typhoon_name'].unique():
            mask = (df['typhoon_name'] == storm) & (df[col].isna())
            if storm in storm_medians:
                df.loc[mask, col] = storm_medians[storm][col]
            else:
                fallback = global_medians[col] if global_medians else df[col].median()
                df.loc[mask, col] = fallback
    return df


def complex_imputation_of_dataset(X_train, X_test, na_cols):
    median_dict = calculate_storm_medians(X_train, na_cols)
    for col in na_cols:
        null_mask_train = X_train[col].isna()
        null_mask_test = X_test[col].isna()
        X_train.loc[null_mask_train, col] = X_train[null_mask_train].apply(
            lambda row: get_storm_median_for_row(row, median_dict, col), axis=1)
        X_test.loc[null_mask_test, col] = X_test[null_mask_test].apply(
            lambda row: get_storm_median_for_row(row, median_dict, col), axis=1)
    return X_train, X_test


def more_complex_imputation_of_dataset(X_train, X_test, na_cols):
    median_dict = calculate_storm_medians(X_train, na_cols)
    for col in na_cols:
        null_mask_train = X_train[col].isna()
        null_mask_test = X_test[col].isna()
        X_train.loc[null_mask_train, col] = X_train[null_mask_train].apply(
            lambda row: get_storm_median_for_row_more_complex(row, median_dict, col), axis=1)
        X_test.loc[null_mask_test, col] = X_test[null_mask_test].apply(
            lambda row: get_storm_median_for_row_more_complex(row, median_dict, col), axis=1)
    return X_train, X_test


# feature selection / cleaning

def drop_correlated_features(X_train, y_train, method='spearman', thresh=0.75):
    '''
    Drops one of each highly correlated pair, keeping whichever has the
    higher correlation with the target.
    '''
    corr_cols_to_drop = []
    cols_to_skip = []
    for i, col in enumerate(X_train.columns):
        for j in X_train.columns[i+1:]:
            if j in cols_to_skip:
                continue
            corr = X_train[[col, j]].corr(method=method).iloc[0, 1]
            if abs(corr) > thresh:
                col_corr = X_train[col].corr(y_train.squeeze(), method=method)
                j_corr = X_train[j].corr(y_train.squeeze(), method=method)
                if abs(col_corr) < abs(j_corr):
                    corr_cols_to_drop.append(col)
                    break
                else:
                    corr_cols_to_drop.append(j)
                    cols_to_skip.append(j)
    return corr_cols_to_drop


def find_correlated_groups(X, method='spearman', thresh=0.75):
    corr = X.corr(method=method).abs()
    G = nx.Graph()
    G.add_nodes_from(X.columns)
    for i in range(len(corr.columns)):
        for j in range(i+1, len(corr.columns)):
            if corr.iloc[i, j] > thresh:
                G.add_edge(corr.columns[i], corr.columns[j])
    return [list(c) for c in nx.connected_components(G) if len(c) > 1]


def select_representative(group, X_train, y_train, method='spearman'):
    correlations = {col: abs(X_train[col].corr(y_train, method=method)) for col in group}
    return max(correlations, key=correlations.get)


def select_best_features(X_train, y_train, groups, model,
                         method='spearman', large_group_thresh=5, cv=5,
                         must_include=None, max_combos=200):
    must_include = must_include or []
    if isinstance(must_include, str):
        must_include = [must_include]

    all_corr_cols = [col for group in groups for col in group]
    base_cols = [col for col in X_train.columns if col not in all_corr_cols]

    for col in must_include:
        if col not in base_cols and col not in all_corr_cols:
            print(f"Warning: '{col}' not found in X_train columns — skipping")

    selected_from_groups = []
    small_groups = []

    for group in groups:
        kept = [col for col in must_include if col in group]
        if len(group) > large_group_thresh:
            best_col = kept[0] if kept else select_representative(group, X_train, y_train, method)
            print(f"Large group ({len(group)} features) → {'kept' if kept else 'selected'}: {best_col}")
            selected_from_groups.append([best_col])
        else:
            if kept:
                group = [col for col in group if col not in kept]
                print(f"Pinning {kept} from small group, searching remaining {len(group)} cols")
                selected_from_groups.append(kept)
                if group:
                    small_groups.append(group)
            else:
                small_groups.append(group)

    if small_groups:
        combos = list(product(*small_groups))
        if len(combos) > max_combos:
            import random
            random.seed(42)
            combos = random.sample(combos, max_combos)
            print(f"Subsampled to {max_combos} combinations")

        print(f"Searching {len(combos)} combinations from {len(small_groups)} small groups...")
        best_score = -np.inf
        best_combo = None
        fixed_cols = base_cols + [col for group in selected_from_groups for col in group]

        for combo in combos:
            cols = fixed_cols + list(combo)
            scores = cross_val_score(model, X_train[cols], y_train, cv=cv, scoring='r2')
            if scores.mean() > best_score:
                best_score = scores.mean()
                best_combo = combo

        print(f"Best exhaustive combo score: {best_score:.4f}")
        selected_from_groups.append(list(best_combo))

    final_cols = base_cols + [col for group in selected_from_groups for col in group]
    for col in must_include:
        if col not in final_cols and col in X_train.columns:
            print(f"Safety net: force-adding '{col}'")
            final_cols.append(col)

    print(f"\nFinal feature count: {len(final_cols)} from original {len(X_train.columns)}")
    return final_cols


def clean_and_split_data(df, target_cols, main_target, filling_method='complex',
                         remove_corr_cols=False, thresh=0.75, selected_storm='None'):
    y = df[main_target]
    X = df.drop(columns=target_cols)

    # drop columns with >95% nulls
    nas_finder = df.isna().sum()
    too_null_cols = [i for i in nas_finder[nas_finder > 0].index if nas_finder.loc[i] / df.shape[0] > 0.95]
    X.drop(columns=too_null_cols, inplace=True)

    # add NA indicator columns
    for i in X.columns[X.isna().any()]:
        X[f'{i}_isna'] = X[i].isna().astype(int)

    # remove storms with no values in any NA column
    na_cols = get_na_columns(X)
    bad_storms = find_bad_storms(X, na_cols)
    mask = ~X['typhoon_name'].isin(bad_storms)
    X, y = X[mask], y[mask]

    # drop single-value columns
    cols_to_drop = [col for col in X.columns if X[col].nunique() == 1]
    X.drop(columns=cols_to_drop, inplace=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    na_cols = get_na_columns(X_train)

    if filling_method == 'simple':
        col_medians = X_train[na_cols].median()
        X_train[na_cols] = X_train[na_cols].fillna(col_medians)
        X_test[na_cols] = X_test[na_cols].fillna(col_medians)
    elif filling_method == 'complex':
        global_medians = X_train[na_cols].median().to_dict()
        X_train, X_test = complex_imputation_of_dataset(X_train, X_test, na_cols)
        X_test[na_cols] = X_test[na_cols].fillna(global_medians)
    elif filling_method == 'more_complex':
        X_train, X_test = more_complex_imputation_of_dataset(X_train, X_test, na_cols)

    if selected_storm != 'None':
        X_test = X_test[X_test['typhoon_name'] == selected_storm]
        y_test = y_test[X_test.index]
        X_train = X_train[X_train['typhoon_name'] == selected_storm]
        y_train = y_train[X_train.index]

    typhoon_train = X_train['typhoon_name'].copy()
    typhoon_test = X_test['typhoon_name'].copy()
    X_train.drop(columns=['typhoon_name'], inplace=True)
    X_test.drop(columns=['typhoon_name'], inplace=True)

    if remove_corr_cols:
        aff_cols = [c for c in X_train.columns if 'aff' in c]
        corr_cols_to_drop = drop_correlated_features(X_train.drop(columns=aff_cols), y_train, 'spearman', thresh)
        X_train.drop(columns=corr_cols_to_drop, inplace=True)
        X_test.drop(columns=corr_cols_to_drop, inplace=True)

    return X_train, X_test, y_train, y_test, typhoon_train, typhoon_test


# model evaluation

def evaluate(model, X_tr, X_te, y_tr, y_te, cv, log=False):
    scores = cross_val_score(model, X_tr, y_tr, cv=cv, scoring='r2')
    cv_r2 = scores.mean()
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)
    if log:
        preds = np.expm1(preds)
        y_te = np.expm1(y_te)
    return cv_r2, r2_score(y_te, preds)


def is_improvement(cv_r2, test_r2, best_score, best_gap, max_gap):
    gap = abs(cv_r2 - test_r2)
    return test_r2 > best_score and gap <= max_gap #returns a bool


def print_step(step, cv_r2, test_r2, accepted, note=""):
    status = "accepted" if accepted else "rejected"
    gap = abs(cv_r2 - test_r2)
    print(f"\n[{step}] {status} {note}")
    print(f"  CV R²: {cv_r2:.4f} | Test R²: {test_r2:.4f} | Gap: {gap:.4f}")


def print_consistent_results(model, X_train, y_train, X_test, y_test, cv=5):
    '''
    Prints CV R²/RMSE (mean ± std) and test set R²/RMSE for consistent
    comparison across models.
    '''
    r2_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='r2')
    rmse_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='neg_root_mean_squared_error')

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"Cross-validation R²: {r2_scores.mean():.4f} ± {r2_scores.std():.4f}")
    print(f"Cross-validation RMSE: {(-rmse_scores.mean()):.4f} ± {rmse_scores.std():.4f}")
    print(f"Test Set R²: {r2_score(y_test, y_pred):.4f}")
    print(f"Test Set RMSE: {root_mean_squared_error(y_test, y_pred):.4f}")


def try_out_models_for_col_thresh(X_train, X_test, y_train, y_test, model, col, thresh=0.2):
    train_mask = X_train[col] >= thresh
    test_mask = X_test[col] >= thresh
    X_train, y_train = X_train[train_mask], y_train[train_mask]
    X_test, y_test = X_test[test_mask], y_test[test_mask]

    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    print(f"mean CV score: {scores.mean()} with thresh: {thresh}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"R²: {r2_score(y_test, y_pred):.4f}  RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")


def auto_optimize_model(
    model,
    X_train, X_test, y_train, y_test,
    groups=None,
    do_gridsearch=False,
    param_grid=None,
    cv=5,
    method='spearman',
    large_group_thresh=5,
    max_combos=200,
    must_include=None,
    max_gap=0.1,
    name_add='',
):
    best_score = -np.inf
    best_gap = np.inf
    best_config = {
        "feature_selection": False,
        "scaling":           None,
        "log_transform":     False,
        "gridsearch":        False,
        "cols":              X_train.columns.tolist(),
        "params":            None,
    }

    X_tr, X_te = X_train.copy(), X_test.copy()
    y_tr, y_te = y_train.copy(), y_test.copy()

    # step 1: baseline
    print("STEP 1: Baseline")
    print("-" * 60)
    cv_r2, test_r2 = evaluate(copy.deepcopy(model), X_tr, X_te, y_tr, y_te, cv=cv)
    best_score = test_r2
    best_gap = abs(cv_r2 - test_r2)
    print(f"  CV R²: {cv_r2:.4f} | Test R²: {test_r2:.4f} | Gap: {best_gap:.4f}")
    best_X_tr, best_X_te = X_tr.copy(), X_te.copy()
    best_y_tr, best_y_te = y_tr.copy(), y_te.copy()

    # step 2: feature selection
    print("\n")
    print("STEP 2: Feature selection")
    print("-" * 60)
    if groups is None:
        groups = find_correlated_groups(X_tr, method=method)

    selected_cols = select_best_features(
        X_tr, y_tr, groups, copy.deepcopy(model),
        method=method, large_group_thresh=large_group_thresh,
        cv=cv, must_include=must_include, max_combos=max_combos,
    )
    X_tr_fs, X_te_fs = X_tr[selected_cols], X_te[selected_cols]
    cv_r2_fs, test_r2_fs = evaluate(copy.deepcopy(model), X_tr_fs, X_te_fs, y_tr, y_te, cv=cv)
    accepted = is_improvement(cv_r2_fs, test_r2_fs, best_score, best_gap, max_gap)
    print_step("Feature selection", cv_r2_fs, test_r2_fs, accepted, f"({len(selected_cols)} cols from {len(X_tr.columns)})")

    model_name = type(model).__name__.lower().replace("regressor", "").replace("classifier", "")
    if accepted:
        pd.Series(selected_cols).to_csv(f'selected_{model_name}cols_lc.csv', index=False)
        best_score = test_r2_fs
        best_gap = abs(cv_r2_fs - test_r2_fs)
        best_X_tr, best_X_te = X_tr_fs.copy(), X_te_fs.copy()
        best_config["feature_selection"] = True
        best_config["cols"] = selected_cols
    else:
        best_X_tr, best_X_te = X_tr.copy(), X_te.copy()

    # step 3: feature scaling
    print("\n")
    print("STEP 3: Feature scaling")
    print("-" * 60)
    for scaler_name, scaler in [("StandardScaler", StandardScaler()), ("MinMaxScaler", MinMaxScaler())]:
        pipeline = make_pipeline(scaler, copy.deepcopy(model))
        cv_r2_sc, test_r2_sc = evaluate(pipeline, best_X_tr, best_X_te, best_y_tr, best_y_te, cv=cv)
        accepted = is_improvement(cv_r2_sc, test_r2_sc, best_score, best_gap, max_gap)
        print_step(f"Scaling ({scaler_name})", cv_r2_sc, test_r2_sc, accepted)
        if accepted:
            best_score = test_r2_sc
            best_gap = abs(cv_r2_sc - test_r2_sc)
            best_config["scaling"] = scaler_name

    # step 4: log transform target
    print("\n")
    print("STEP 4: Log transform target")
    print("-" * 60)
    if (best_y_tr >= 0).all():
        y_tr_log = np.log1p(best_y_tr)
        y_te_log = np.log1p(best_y_te)
        if best_config["scaling"] == "StandardScaler":
            m_log = make_pipeline(StandardScaler(), copy.deepcopy(model))
        elif best_config["scaling"] == "MinMaxScaler":
            m_log = make_pipeline(MinMaxScaler(), copy.deepcopy(model))
        else:
            m_log = copy.deepcopy(model)
        cv_r2_log, test_r2_log = evaluate(m_log, best_X_tr, best_X_te, y_tr_log, y_te_log, log=True, cv=cv)
        accepted = is_improvement(cv_r2_log, test_r2_log, best_score, best_gap, max_gap)
        print_step("Log transform target", cv_r2_log, test_r2_log, accepted)
        if accepted:
            best_score = test_r2_log
            best_gap = abs(cv_r2_log - test_r2_log)
            best_y_tr, best_y_te = y_tr_log, y_te_log
            best_config["log_transform"] = True
    else:
        print("  Skipped — target contains negative values")

    # step 5: gridsearch
    print("\n")
    print("STEP 5: Gridsearch")
    print("-" * 60)
    if do_gridsearch and param_grid:
        model_step_name = type(model).__name__.lower()
        if best_config["scaling"] == "StandardScaler":
            base = make_pipeline(StandardScaler(), copy.deepcopy(model))
            prefixed_grid = {f"{model_step_name}__{k}": v for k, v in param_grid.items()}
        elif best_config["scaling"] == "MinMaxScaler":
            base = make_pipeline(MinMaxScaler(), copy.deepcopy(model))
            prefixed_grid = {f"{model_step_name}__{k}": v for k, v in param_grid.items()}
        else:
            base = copy.deepcopy(model)
            prefixed_grid = param_grid

        gs = RandomizedSearchCV(base, prefixed_grid, n_iter=100, cv=cv, scoring='r2',
                                random_state=42, n_jobs=-1, verbose=1)
        gs.fit(best_X_tr, best_y_tr)

        preds = gs.best_estimator_.predict(best_X_te)
        if best_config["log_transform"]:
            preds = np.expm1(preds)
            yte_ = np.expm1(best_y_te)
        else:
            yte_ = best_y_te

        test_r2_gs = r2_score(yte_, preds)
        accepted = is_improvement(gs.best_score_, test_r2_gs, best_score, best_gap, max_gap)
        print_step("Gridsearch", gs.best_score_, test_r2_gs, accepted, f"best params: {gs.best_params_}")

        if accepted:
            best_score = test_r2_gs
            best_gap = abs(gs.best_score_ - test_r2_gs)
            best_config["gridsearch"] = True
            best_config["params"] = gs.best_params_
    else:
        print("Skipped — do_gridsearch=False or no param_grid provided")

    # build final model and save
    model_name_parts = [type(model).__name__.lower().replace("regressor", "").replace("classifier", "")]
    if best_config["scaling"] == "StandardScaler":
        model_name_parts.append("standard")
    elif best_config["scaling"] == "MinMaxScaler":
        model_name_parts.append("minmax")
    if best_config["log_transform"]:
        model_name_parts.append("log")
    if best_config["feature_selection"]:
        model_name_parts.append("reduced_cols")
    if best_config["gridsearch"]:
        model_name_parts.append("tuned")
    model_name = "_".join(model_name_parts)

    if best_config["scaling"] == "StandardScaler":
        final_model = make_pipeline(StandardScaler(), copy.deepcopy(model))
    elif best_config["scaling"] == "MinMaxScaler":
        final_model = make_pipeline(MinMaxScaler(), copy.deepcopy(model))
    else:
        final_model = copy.deepcopy(model)

    if best_config["gridsearch"] and best_config["params"]:
        final_model.set_params(**best_config["params"])

    final_model.fit(best_X_tr, best_y_tr)

    os.makedirs("../models", exist_ok=True)
    pickle_path = f"../models/{model_name}_{name_add}.pkl"
    with open(pickle_path, "wb") as f:
        pickle.dump(final_model, f)

    best_config["model_name"] = model_name
    best_config["pickle_path"] = pickle_path

    print("\n")
    print("FINAL SUMMARY")
    print("-" * 60)
    print(f"Model name: {model_name}")
    print(f"Best Test R²: {best_score:.4f}")
    print(f"Best Gap: {best_gap:.4f}")
    print(f"Feature selection: {best_config['feature_selection']}")
    print(f"Scaling: {best_config['scaling']}")
    print(f"Log transform: {best_config['log_transform']}")
    print(f"Gridsearch: {best_config['gridsearch']}")
    if best_config["params"]:
        print(f"Best params: {best_config['params']}")
    print(f"Final feature count:{len(best_config['cols'])}")
    print(f"Saved to: {pickle_path}")

    return best_config, best_X_tr, best_X_te, best_y_tr, best_y_te, final_model


# residuals

def compute_residuals(model, X_test, y_test):
    '''
    Returns residuals (actual - predicted) and predictions as a tuple.
    '''
    y_pred = model.predict(X_test)
    return y_test.values.flatten() - y_pred, y_pred


def plot_residuals(model, X_test, y_test, model_name, figure_name):
    '''
    Plots residuals vs predicted values and saves the figure.
    '''
    residuals, y_pred = compute_residuals(model, X_test, y_test)
    plt.figure(figsize=(10, 4))
    plt.scatter(y_pred, residuals, alpha=0.3)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel("Predicted")
    plt.ylabel("Residual")
    plt.title(f"Residual Plot for {model_name} Model")
    plt.tight_layout()
    plt.savefig(f'../figures/{figure_name}')
    plt.show()


# feature importance

def print_feature_importance(model, feature_names, model_name, best_color, other_color):
    '''
    Plots MDI feature importance with per-tree std. 
    For boosting models use print_feature_importance_pm instead.
    '''
    importances = model.feature_importances_
    std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
    df_imp = pd.DataFrame({
        'importance': importances,
        'std': std,
        'feature': feature_names
    }).sort_values('importance', ascending=True)
    final_df = df_imp[df_imp['importance'] > 0.001]
    colors = [other_color] * (final_df.shape[0] - 1) + [best_color]
    fig, ax = plt.subplots(figsize=(10, len(feature_names) * 0.25))
    ax.barh(final_df['feature'], final_df['importance'], xerr=final_df['std'], align='center', color=colors)
    ax.set_title(f"Feature importances in {model_name} Model")
    ax.set_xlabel("Mean decrease in impurity")
    fig.tight_layout()
    plt.show()


def print_feature_importance_pm(model, X_val, y_val, feature_names, model_name, other_color, best_color, n_repeats=30):
    '''
    Plots permutation importance on a validation set. Preferred over MDI for boosting models.
    '''
    r = permutation_importance(model, X_val, y_val, n_repeats=n_repeats, random_state=42)
    df_imp = pd.DataFrame({
        'importance': r.importances_mean,
        'std': r.importances_std,
        'feature': feature_names
    }).sort_values('importance', ascending=True)
    final_df = df_imp[df_imp['importance'] > 0.001].tail(20)
    colors = [other_color] * (final_df.shape[0] - 1) + [best_color]
    fig, ax = plt.subplots(figsize=(10, len(final_df) * 0.35))
    ax.barh(final_df['feature'], final_df['importance'], xerr=final_df['std'], align='center', color=colors)
    ax.set_title(f"Feature importances for {model_name} Model using Permutation Importance")
    ax.set_xlabel("Mean accuracy decrease")
    fig.tight_layout()
    plt.show()


# plotting helpers

def get_better_dists_for_hists(col, n_bins):
    '''
    Returns evenly sized quantile-based buckets for a column.
    '''
    all_limits = [col.quantile((1/n_bins)*i) for i in range(1, n_bins+1)]
    steps_dict = {}
    lower = f'{(0.0 * 100):.2f}%'
    for i in all_limits:
        upper = f'{i*100:.2f}%'
        if i == all_limits[-1]:
            bound_name = f'{lower}+'
        elif i == all_limits[0]:
            bound_name = f'<{upper}'
        else:
            bound_name = f'{lower} - {upper}'
        steps_dict[i] = bound_name
        lower = f'{(i) * 100:.2f}%'
    return steps_dict


def box_availability_rates(row, hist_box_dict):
    for key in sorted(hist_box_dict.keys()):
        if row <= key:
            return hist_box_dict[key]


def plot_hisplots_for_subgroups(data, group_col, hist_col, figsize=(15, 15),
                                gridspec_row=4, gridspec_col=7, color_list=True,
                                colors=[], group_type='NTL Data Availability',
                                hist_column_type='Residuals', model_name='ExtraTrees', complex=False):
    '''
    Plots a gridspec of histplots for each group in group_col to compare distributions.
    '''
    if complex:
        data['ground_truth'] = hist_col
        hist_col = 'ground_truth'
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(gridspec_row, gridspec_col)
    for i, group in enumerate(list(data[group_col].unique())):
        color = colors[i] if color_list else colors
        ax = plt.subplot(gs[i])
        df = data[data[group_col] == group]
        sns.histplot(data=df, x=hist_col, color=color, kde=True, stat="density", linewidth=0, bins=50)
        ax.set_xlabel('')
        ax.set_title(f"Availability Group: {group}")
        ax.grid(True)
    fig.suptitle(f'Distribution of Model {hist_column_type} by {group_type}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'../figures/residual_gridspec_distribution_{model_name}.png')
    plt.show()


def get_highest_corrs_with_target_plot(X, y, data_name):
    corr = X.corrwith(y, method='spearman').to_frame(name='Spearman Correlation').dropna()
    corr = corr[abs(corr) > 0.07].dropna().sort_values('Spearman Correlation', ascending=False)

    fig, ax = plt.subplots(figsize=(6, len(corr) * 0.4))
    fig.patch.set_facecolor(BACKGROUND_COLOR)
    ax.set_facecolor(BACKGROUND_COLOR)

    sns.heatmap(corr, annot=False, cmap=thesis_cmap_div, center=0, vmin=-1, vmax=1,
                linewidths=0.5, xticklabels=False, yticklabels=False, ax=ax)

    for i, col in enumerate(corr.index):
        val = corr.loc[col, 'Spearman Correlation']
        text_color = 'black' if abs(val) < 0.6 else 'white'
        ax.text(0.02, i + 0.5, col, ha='left', va='center', fontsize=8, color=text_color)
        ax.text(0.98, i + 0.5, f"{val:.2f}", ha='right', va='center', fontsize=8,
                fontweight='bold', color=text_color)

    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.tight_layout()


def plot_geodataframe_choropleth(gdf, column, cmap="viridis", figsize=None, title=None,
                                 vmin=None, vmax=None, edge_color="none", edge_width=0.1,
                                 missing_color="lightgrey"):
    '''
    Plots a choropleth of a GeoDataFrame column using polygon geometry.
    '''
    if figsize is None:
        bounds = gdf.total_bounds
        aspect = (bounds[2] - bounds[0]) / (bounds[3] - bounds[1]) if (bounds[3] - bounds[1]) > 0 else 1.5
        figsize = (10 * aspect, 10)

    fig, ax = plt.subplots(figsize=figsize)
    gdf.plot(column=column, ax=ax, cmap=cmap, vmin=vmin, vmax=vmax,
             edgecolor=edge_color, linewidth=edge_width,
             missing_kwds={"color": missing_color, "label": "No data"},
             legend=True, legend_kwds={"label": column, "orientation": "vertical", "shrink": 0.7, "pad": 0.02})

    ax.set_title(title or f"Choropleth of '{column}'", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Longitude", fontsize=11)
    ax.set_ylabel("Latitude", fontsize=11)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.2f}"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.2f}"))
    plt.xticks(rotation=45, ha="right")
    ax.set_aspect("equal")
    plt.tight_layout()
    return fig, ax


def plot_choropleth_grid(gdfs, columns, titles=None, cmaps=None, figsize=None, suptitle=None,
                         vmin=None, vmax=None, edge_color="none", edge_width=0.1,
                         missing_color="lightgrey", shared_scale=False, diverging=None, ref_gdf=None):
    '''
    Plots a side-by-side grid of choropleths, one per GeoDataFrame/column pair.
    Pass ref_gdf to fix the map bounds to the full grid extent.
    '''
    n = len(gdfs)
    if len(columns) != n:
        raise ValueError(f"Expected {n} columns, got {len(columns)}")

    titles = titles or columns
    cmaps = cmaps or ["viridis"] * n
    diverging = diverging or [False] * n
    ref_bounds = ref_gdf.total_bounds if ref_gdf is not None else gdfs[0].total_bounds

    if shared_scale:
        all_vals = np.concatenate([gdf[col].dropna().values for gdf, col in zip(gdfs, columns)])
        vmin, vmax = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))

    if figsize is None:
        aspect = (ref_bounds[2] - ref_bounds[0]) / (ref_bounds[3] - ref_bounds[1]) if (ref_bounds[3] - ref_bounds[1]) > 0 else 1.0
        figsize = (7 * aspect * n, 9)

    fig, axes = plt.subplots(1, n, figsize=figsize, layout="constrained")
    if n == 1:
        axes = [axes]

    for gdf, ax, col, title, cmap, is_div in zip(gdfs, axes, columns, titles, cmaps, diverging):
        col_data = gdf[col].dropna()
        if is_div and col_data.min() < 0 and col_data.max() > 0:
            vmax_sym = np.nanpercentile(np.abs(col_data), 95)
            norm = mcolors.TwoSlopeNorm(vmin=-vmax_sym, vcenter=0, vmax=vmax_sym)
        else:
            norm = None

        gdf.plot(column=col, ax=ax, cmap=cmap, norm=norm,
                 vmin=vmin if norm is None else None,
                 vmax=vmax if norm is None else None,
                 edgecolor=edge_color, linewidth=edge_width,
                 missing_kwds={"color": missing_color, "label": "No data"},
                 legend=True, legend_kwds={"orientation": "vertical", "shrink": 0.6, "pad": 0.04, "fraction": 0.03})

        ax.set_xlim(ref_bounds[0], ref_bounds[2])
        ax.set_ylim(ref_bounds[1], ref_bounds[3])
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=15, fontweight="bold", pad=5)
        ax.set_xlabel("Longitude", fontsize=10)
        ax.set_ylabel("Latitude", fontsize=10)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.2f}"))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.2f}"))
        ax.tick_params(axis="x", rotation=45)

    if suptitle:
        fig.suptitle(suptitle, fontsize=17, fontweight="bold")

    return fig, axes