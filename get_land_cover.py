#!/usr/bin/env python
# coding=utf-8
import matplotlib
matplotlib.use('Agg')
import numpy as np
import rasterio
import xarray as xr
import sys
import os 
from osgeo import ogr  
ogr.UseExceptions()
import warnings
from rasterio.crs import CRS
import geopandas as gpd

nc_cache = {}

#%%
NODATA_VALUE = -9999
region = "philippines"

#%%

def _open_nc_tile(base_path, year, tile):
    key = (int(year), tile)
    if key in nc_cache:
        return nc_cache[key]
    nc_path = f"{base_path}/{year}/{tile}_{year}.nc"
    if not os.path.exists(nc_path):
        nc_cache[key] = None
        return None
    ds = xr.open_dataset(nc_path)
    nc_cache[key] = ds
    return ds

def save_raster(output_path, data, ref_ds, xmin, xmax, ymin, ymax, raster_image):
    data_full = np.full((ref_ds.height, ref_ds.width), np.nan)
    mask = raster_image[xmin:xmax, ymin:ymax] == 255
    data_masked = np.full((xmax - xmin, ymax - ymin), np.nan)
    data_masked[mask] = data[mask]
    data_full[xmin:xmax, ymin:ymax] = data_masked
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    profile = ref_ds.profile.copy()
    profile.update({
        'driver': 'GTiff',
        'height': ref_ds.height,
        'width': ref_ds.width,
        'count': 1,
        'dtype': 'float64',
        'compress': 'lzw',
        'transform': ref_ds.transform,
        'crs': CRS.from_epsg(4326),
        'nodata': NODATA_VALUE
    })
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(data_full, 1)

warnings.filterwarnings("ignore")
base_path   = "data/tiles"
output_base = "data/output_new/"
region      = "philippines"
syear       = 2018

dims_to_pull = [
    'ghsl_smod_2010',
    'ghsl_smod_2015',
    'ghsl_smod_2020',
    'ghs_smod_urban_suburban_2010',
    'ghs_smod_urban_suburban_2015',
    'ghs_smod_urban_suburban_2020',
]

# full categorical layers — for smod_all_interp.nc
smod_full_years = {
    'ghsl_smod_2010': 2010,
    'ghsl_smod_2015': 2015,
    'ghsl_smod_2020': 2020,
}

# binary urban/suburban layers — for smod_urban_interp.nc
smod_urban_years = {
    'ghs_smod_urban_suburban_2010': 2010,
    'ghs_smod_urban_suburban_2015': 2015,
    'ghs_smod_urban_suburban_2020': 2020,
}

hazard_tif_path = "data/vectors/philippines.tif"

if os.path.exists(hazard_tif_path):

    nrow      = 4018
    ncol      = 3008
    tile_nrow = 2400
    tile_ncol = 2400

    hazard_shp_path = "data/shape_files/buffered_files/PH_dissolved_buffer.shp"
    gdf = gpd.read_file(hazard_shp_path)
    gdf = gdf.set_crs("EPSG:32651").to_crs("EPSG:4326")

    haz_tif      = rasterio.open(hazard_tif_path)
    raster_image = haz_tif.read(1)

    aff_rows_list, aff_cols_list = np.where(raster_image == 255)
    if len(aff_rows_list) == 0:
        print("No affected polygon")
        sys.exit()

    xmin = min(aff_rows_list)
    xmax = max(aff_rows_list) + 1
    ymin = min(aff_cols_list)
    ymax = max(aff_cols_list) + 1

    nc_ds_ref   = _open_nc_tile(base_path, syear, "h29v06")
    nc_res      = abs(float(nc_ds_ref.x[1]) - float(nc_ds_ref.x[0]))
    nc_origin_x = float(nc_ds_ref.x.min()) - nc_res / 2
    nc_origin_y = float(nc_ds_ref.y.max()) + nc_res / 2

    haz_origin_x = haz_tif.transform.c
    haz_origin_y = haz_tif.transform.f

    row_offset = int((nc_origin_y - haz_origin_y) / nc_res)
    col_offset = int((haz_origin_x - nc_origin_x) / nc_res)
    ymin_nc = ymin + col_offset
    ymax_nc = ymax + col_offset

    minX, minY, maxX, maxY = gdf.total_bounds
    h_min = int((minX + 180) // 10)
    h_max = int((maxX + 180) // 10)
    v_min = int((90 - maxY) // 10)
    v_max = int((90 - minY) // 10)

    rows_aff = xmax - xmin
    cols_aff = ymax - ymin
    xmin_nc  = xmin + row_offset
    xmax_nc  = xmax + row_offset

    # extract all dims and save as TIFs
    for dim_to_pull in dims_to_pull:
        stacked_land = np.full((rows_aff, cols_aff), np.nan)

        for h in range(h_min, h_max + 1):
            for v in range(v_min, v_max + 1):
                j_min   = (h - h_min) * tile_ncol
                i_min   = (v - v_min) * tile_nrow
                j_max   = j_min + tile_ncol
                i_max   = i_min + tile_nrow
                tile_id = "h" + str(h).zfill(2) + "v" + str(v).zfill(2)

                ov_row_start = max(i_min, xmin_nc)
                ov_row_end   = min(i_max, xmax_nc)
                ov_col_start = max(j_min, ymin_nc)
                ov_col_end   = min(j_max, ymax_nc)
                if ov_row_start >= ov_row_end or ov_col_start >= ov_col_end:
                    continue

                stack_row_start = max(ov_row_start - xmin_nc, 0)
                stack_row_end   = min(ov_row_end   - xmin_nc, rows_aff)
                stack_col_start = max(ov_col_start - ymin_nc, 0)
                stack_col_end   = min(ov_col_end   - ymin_nc, cols_aff)
                if stack_row_end <= stack_row_start or stack_col_end <= stack_col_start:
                    continue

                nc_ds = _open_nc_tile(base_path, syear, tile_id)
                if nc_ds is None or dim_to_pull not in nc_ds:
                    print(f"NC file or dim not found for {tile_id} year {syear}")
                    continue

                actual_nrow = nc_ds.sizes["y"]
                actual_ncol = nc_ds.sizes["x"]

                tile_row_start = max(ov_row_start - i_min, 0)
                tile_row_end   = min(ov_row_end   - i_min, actual_nrow)
                tile_col_start = max(ov_col_start - j_min, 0)
                tile_col_end   = min(ov_col_end   - j_min, actual_ncol)

                row_size = min(tile_row_end - tile_row_start, stack_row_end - stack_row_start)
                col_size = min(tile_col_end - tile_col_start, stack_col_end - stack_col_start)
                if row_size <= 0 or col_size <= 0:
                    continue

                tile_row_end        = tile_row_start + row_size
                tile_col_end        = tile_col_start + col_size
                stack_row_end_final = stack_row_start + row_size
                stack_col_end_final = stack_col_start + col_size

                print(f"Processing {dim_to_pull} | tile {tile_id}")
                lc_block = nc_ds[dim_to_pull].isel(
                    y=slice(tile_row_start, tile_row_end),
                    x=slice(tile_col_start, tile_col_end),
                ).values

                stacked_land[
                    stack_row_start:stack_row_end_final,
                    stack_col_start:stack_col_end_final,
                ] = lc_block

        filename_landcover = f"{output_base}/landcover/{region}_{dim_to_pull}.tif"
        save_raster(filename_landcover, stacked_land, haz_tif, xmin, xmax, ymin, ymax, raster_image)
        print(f"Saved {filename_landcover}")

    nc_cache.clear()

    # NetCDF 1: full categorical SMOD interpolation
    year_arrays = {}
    for dim, snap_year in smod_full_years.items():
        tif_path = f"{output_base}/landcover/{region}_{dim}.tif"
        with rasterio.open(tif_path) as src:
            data = src.read(1).astype(float)
            data[data == NODATA_VALUE] = np.nan
        year_arrays[snap_year] = data
        print(f"Loaded {dim} snap year {snap_year}, unique values: {np.unique(data[~np.isnan(data)])}")

    snap_years = sorted(year_arrays.keys())
    data_stack = np.stack([year_arrays[y] for y in snap_years], axis=0)

    ds = xr.Dataset(
        {"smod": (["year", "y", "x"], data_stack)},
        coords={"year": snap_years,
                "y":    np.arange(data_stack.shape[1]),
                "x":    np.arange(data_stack.shape[2])}
    )
    ds_interp = ds.interp(year=np.arange(2010, 2021), method="linear")
    nc_all_path = f"{output_base}/landcover/{region}_smod_all_interp.nc"
    ds_interp.to_netcdf(nc_all_path)
    print(f"Saved full categorical SMOD NC: {nc_all_path}")

    # NetCDF 2: binary urban/suburban mask interpolated
    year_arrays_urban = {}
    for dim, snap_year in smod_urban_years.items():
        tif_path = f"{output_base}/landcover/{region}_{dim}.tif"
        with rasterio.open(tif_path) as src:
            data = src.read(1).astype(float)
            data[data == NODATA_VALUE] = np.nan
        year_arrays_urban[snap_year] = data
        print(f"Loaded {dim} snap year {snap_year}")

    snap_years_urban = sorted(year_arrays_urban.keys())
    data_stack_urban = np.stack([year_arrays_urban[y] for y in snap_years_urban], axis=0)

    ds_urban = xr.Dataset(
        {"smod_urban": (["year", "y", "x"], data_stack_urban)},
        coords={"year": snap_years_urban,
                "y":    np.arange(data_stack_urban.shape[1]),
                "x":    np.arange(data_stack_urban.shape[2])}
    )
    ds_urban_interp = ds_urban.interp(year=np.arange(2010, 2021), method="linear")
    nc_urban_path = f"{output_base}/landcover/{region}_smod_urban_interp.nc"
    ds_urban_interp.to_netcdf(nc_urban_path)
    print(f"Saved urban-only SMOD NC: {nc_urban_path}")

else:
    print(f"{hazard_tif_path} does not exist")

for _ds in nc_cache.values():
    if _ds is not None:
        _ds.close()