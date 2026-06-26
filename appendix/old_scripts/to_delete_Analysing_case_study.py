#!/usr/bin/env python
# coding=utf-8

import matplotlib
matplotlib.use('Agg')
import numpy as np
import rasterio
import sys
import os 
from osgeo import ogr  
ogr.UseExceptions()
from datetime import datetime, timedelta
import warnings
import shapefile
import pandas as pd
import csv
from rasterio.crs import CRS
from shapely.geometry import shape
from shapely import wkt
import geopandas as gpd
from shapely.geometry import Point
from shapely.strtree import STRtree
from shapely.geometry import box
import psutil
import shutil
from pathlib import Path
import re

def is_leap_year(year):
    """ Check if a year is a leap year """
    return (year % 400 == 0) or ((year % 4 == 0) and (year % 100 != 0))

def save_raster(output_path, data, ref_ds, im_row, im_col, xmin, xmax, ymin, ymax, raster_image):

    #data_full = np.zeros((im_row,im_col))
    #data_full[xmin:xmax, ymin:ymax] = data
    #data_full[raster_image!=255] = np.nan #Mask out unaffected area

    data_full = np.full((im_row, im_col), np.nan)  # Start with all NaNs

    # Only update affected area
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

def find_persistent_recovery(ts_recovered, start_idx, total_days, min_ratio, window, min_valid):
    persist_checked = False

    for t in range(start_idx + window, total_days):
        window_vals = ts_recovered[t - window : t]
        valid = window_vals[~np.isnan(window_vals)]
        if len(valid) >= min_valid and np.nansum(valid) / len(valid) >= min_ratio:
            persist_checked=True
            for idx_in_window, val in enumerate(window_vals):
                if val == 1:
                    return (t - window) + idx_in_window, persist_checked
    return None, persist_checked

#%%

#%%

BUFFER_pre = 150 
BUFFER_post = 150
NODATA_VALUE = -9999
RECOVERY_NOT_FOUND = 999
NOT_ENOUGH_DATA = 888
# NEW: availability threshold for postimpact/postworst and availability-adjusted durations
AVAIL_THRESHOLDS = [0.3, 0.5]

RECALC=True 
store_for_print = False
region = "dominica"
MEM_thresh = 15

sdate = pd.to_datetime("2017-09-18") #when Maria made landfall in Dominica
edate = pd.to_datetime("2017-09-19") 

def check_memory(threshold_gb=MEM_thresh):
    process = psutil.Process(os.getpid())
    mem_used = process.memory_info().rss / (1024 ** 3)  # Memory in GB
    if mem_used > threshold_gb:
        print(f"Memory usage too high ({mem_used:.2f} GB), exceeding threshold of {threshold_gb} GB.")

        sys.exit(2)

#%%

from pathlib import Path

def _parse_time_list(val):
    """Parse list-like field into tz-naive pandas Timestamps."""
    if isinstance(val, list):
        items = val
    elif isinstance(val, str):
        try:
            parsed = ast.literal_eval(val)
            items = parsed if isinstance(parsed, list) else [parsed]
        except Exception:
            items = re.findall(r"\d{4}-\d{2}-\d{2}[^,\]]*", val)
    else:
        items = []
    return [pd.to_datetime(x).tz_localize(None) for x in items if x]

#%%


s_t = 3

warnings.filterwarnings("ignore")
base_path = "/gpfs/work4/0/FWC2/MYRIAD/data/NTL_data/"
hazard_base = f"{base_path}Processed_data/Hazard_geometries/{region}/" 

output_base = f"{base_path}/Results/{region}/"
post_base = f"{base_path}/Results/{region}/postprocessing"

# Add threshold subdirectory if needed  
if s_t != 3:
    output_base = f"{output_base}/threshold_{s_t}"

hazard_tif_path = f"{hazard_base}/rasters/{region}.tif"

if os.path.exists(hazard_tif_path): 

    nrow = 2400
    ncol = 2400            

    hazard_shp_path = f"{hazard_base}separate_polygons/{region}.shp"

    haz_shp = shapefile.Reader(hazard_shp_path)  
    shapes = haz_shp.shapes()
    
    #Skip previously processed files
    filename_check = f"{output_base}/impact_duration/{region}_impact_dur.tif"
    if os.path.exists (filename_check) and RECALC==False:
        print ("skipped because already calculated")
        sys.exit() #stop this run
    
    print (f"Start analysing for {region}")

    # Define the start and end time indices
    
    haz_tif = rasterio.open(hazard_tif_path)
    raster_image = haz_tif.read(1)

    aff_rows_list, aff_cols_list = np.where((raster_image == 255)) #255 indicates affected area
    
    if len(aff_rows_list) == 0:
      
        print ("No affected polygon")
        sys.exit()  # stop this run

    rows_aff = max(aff_rows_list) - min(aff_rows_list)+1 #total n affected rows
    cols_aff = max(aff_cols_list) - min(aff_cols_list)+1 #total n affected cols
    xmin = min(aff_rows_list)
    xmax = max(aff_rows_list) + 1
    ymin = min(aff_cols_list)
    ymax = max(aff_cols_list) + 1
    
    check_memory()  

    #date calculation
    months=[0,31,59,90,120,151,181,212,243,273,304,334] #Months expressed in DOY
    kx=len(shapes) #should be just 1
    
    print (f"Length shapes: {kx}")
    minX, minY, maxX, maxY = shapes[0].bbox 

    h_min=int((minX + 180)//10)
    h_max=int((maxX + 180)//10)
    v_min=int((90 - maxY)//10)
    v_max=int((90 - minY)//10)
    
    deltah = h_max - h_min+1
    deltav = v_max - v_min+1
    im_row = int(deltav * nrow)
    im_col = int(deltah * ncol)            
    
    #Extent analysis window to include potential signals from evacuations or delayed impacts
    sdate = sdate - pd.Timedelta(days=3) #Start 3 days before event start
    edate = edate + pd.Timedelta(days=5) #End 5 days after event end

    syear, smonth, sday = sdate.year, sdate.month, sdate.day
    eyear, emonth, eday = edate.year, edate.month, edate.day

    #Caculate start and end dates in DOY
    sd_doy = months[smonth-1] + sday #start from 1
    ed_doy = months[emonth-1] + eday

    #Ensure accuracy for leap years
    idx = 0
    if is_leap_year(syear):
        idx = 1 #leap year
    if (idx==1) and (smonth > 2):
        sd_doy += 1 
    idx = 0
    days_in_yr = 365
    if is_leap_year(eyear):
        idx = 1
        days_in_yr = 366
    if (idx == 1) and (emonth > 2):
        ed_doy += 1
    
    duration_event = (edate - sdate).days + 1 #general statement that works under any circumstance
    total_days_analysis = duration_event + BUFFER_pre + BUFFER_post 

    def estimate_memory_usage_gb(days, rows, cols, dtype='float64'):
        dtype_size = np.dtype(dtype).itemsize
        total_bytes = days * rows * cols * dtype_size
        total_gb = total_bytes / (1024 ** 3)
        return total_gb

    # Estimate memory before allocation
    
    estimated_gb = estimate_memory_usage_gb(total_days_analysis, rows_aff, cols_aff)

    if estimated_gb > MEM_thresh:
        print(f"Estimated memory usage ({estimated_gb:.2f} GB) exceeds {MEM_thresh} GB.")

        sys.exit()
    
    stacked_NTL = np.zeros((total_days_analysis, rows_aff, cols_aff)) #Initialise
    analysis_start_date = sdate - timedelta(days=BUFFER_pre)
    analysis_end_date = edate + timedelta(days=BUFFER_post)

    #Stack relevant NTL data, looping over DOYs
    #the d_i is in the range 0 +150 (150 being the event start) + duration (e.g., 3)+ 150 (e.g., 303)
    for d_i in range(0, total_days_analysis):        

        check_memory()  

        analysis_date = analysis_start_date + timedelta(days=d_i)
        year1 = analysis_date.year
        doy_i = analysis_date.timetuple().tm_yday

        #Collect relevant NTL tiles for this DOY
        image_NTL = np.zeros((im_row,im_col))

        for h in range(h_min, h_max + 1):
            for v in range(v_min, v_max + 1):
                j_min = (h-h_min) * ncol
                i_min = (v-v_min) * nrow
                j_max = j_min + ncol
                i_max = i_min + nrow          
                str2="h"+str(h).zfill(2)+"v"+str(v).zfill(2)
                NTL_filepath = f"{base_path}/Processed_data/Black_Marble_processed/corrected_NTL/"+str(year1)+"/"+str2+"/"+str(doy_i).zfill(3)+".tif"

                if os.path.exists(NTL_filepath):
                    try:
                        ds = rasterio.open(NTL_filepath)
                        NTL = ds.read(1)
                        image_NTL[i_min:i_max,j_min:j_max] = NTL
                        check_memory()  
 
                    except Exception:
                        print (f"Cannot open NTL file {NTL_filepath}")
                        image_NTL[i_min:i_max,j_min:j_max]=np.nan
                else:
                    image_NTL[i_min:i_max,j_min:j_max]=np.nan

        check_memory()  
        stacked_NTL[d_i,:,:] = image_NTL[xmin:xmax,ymin:ymax].copy()
        check_memory()  


    #0. Calculate statistics
    img_NTL_mean = np.nanmean(stacked_NTL, axis=0)
    img_NTL_std = np.nanstd(stacked_NTL, axis=0)

    #Define outliers
    img_min = img_NTL_mean - s_t * img_NTL_std
    img_max = img_NTL_mean + s_t * img_NTL_std

    #Remove outliers in pre-event period
    pre_event_NTL = stacked_NTL[0:BUFFER_pre,:,:]
    for dN in range(0, BUFFER_pre):
        Np = pre_event_NTL[dN,:,:]
        Np[Np>img_max] = np.nan
        Np[Np<img_min] = np.nan

    #Remove any pixels with very low pre-event data because no possibility to make threshold
    pre_event_data = stacked_NTL[0:BUFFER_pre,:,:].copy()
    valid_data = np.zeros((BUFFER_pre, rows_aff, cols_aff))
    valid_data[pre_event_data==pre_event_data] = 1 #True for all non-nan cells
    valid_counts = np.nansum(valid_data,axis=0)
    hr, hc = np.where(valid_counts < 5) #Removing any pixels with less than 5 values over the pre-event period
    stacked_NTL[:,hr,hc] = np.nan
    
    #1. Uncertainty---available pixel per ratio of avalable pixels during flooding period
    check_memory()  

    idx_event_start = BUFFER_pre #set startdate to the analysis period
    idx_event_end = BUFFER_pre + duration_event #set enddate to buffer period + duration 
    
    D1 = stacked_NTL[idx_event_start:idx_event_end,:,:].copy() #Find data from during event
    D2 = np.zeros((duration_event, rows_aff, cols_aff)) 
    D2[D1==D1]=1
    D3 = np.nansum(D2,axis=0) #Find valid observations
    avail_perc = D3 / duration_event #Calculate ratio of valid observations vs. total observations

    if np.all(avail_perc == 0):
        print ("Not enough data")        
        sys.exit()  # stop this run

    filename_uncertainty = output_base + "/available_percentage/" + region + "_available" + ".tif"
    save_raster(filename_uncertainty, avail_perc, haz_tif, im_row, im_col,xmin, xmax, ymin, ymax, raster_image)

    print("done first step")
        # Uncertainty finish
        
    #2. DA&DP
    check_memory()  


    pre_event_mean = np.nanmean(stacked_NTL[0:int(BUFFER_pre),:,:],axis=0) #pre event        
    pre_event_std = np.nanstd(stacked_NTL[0:BUFFER_pre,:,:],axis=0) #pre event
    min_NTL_affected = np.nanmin(stacked_NTL[idx_event_start:idx_event_end,:,:],axis=0) #during event
        
    filename_std = output_base +"/pre_std/"+region+ "_pre_event_std" + ".tif"
    save_raster(filename_std, pre_event_std, haz_tif, im_row, im_col,xmin, xmax, ymin, ymax, raster_image)

    filename_mean = output_base +"/pre_mean/"+region+ "_pre_event_mean" + ".tif"
    save_raster(filename_mean, pre_event_mean, haz_tif, im_row, im_col,xmin, xmax, ymin, ymax, raster_image)

    filename_min = output_base +"/event_min/"+region + "_min_NTL_during_event" + ".tif"
    save_raster(filename_min, min_NTL_affected, haz_tif, im_row, im_col,xmin, xmax, ymin, ymax, raster_image)

    """ This is calculating the max DP and DA using the min_NTL during the event, not NTL value at day i"""
    with np.errstate(divide='ignore', invalid='ignore'):
        DAmax = (pre_event_mean - min_NTL_affected) / pre_event_std   
        DAmax[~np.isfinite(DAmax)] = np.nan          
        DPmax = (pre_event_mean - min_NTL_affected) / pre_event_mean  
        DPmax[~np.isfinite(DPmax)] = np.nan                              
    
    DAmax[(DAmax > 999) | (DAmax < -999)] = np.nan
    DPmax[(DPmax > 999) | (DPmax < -999)] = np.nan

    #DAmax and DPmax are during event, based on min_NTL_affected
    filename_DA = output_base +"/DAmax_150/"+ region + "_DAmax" + ".tif"
    save_raster(filename_DA, DAmax, haz_tif, im_row, im_col,xmin, xmax, ymin, ymax, raster_image)
    
    filename_DP = output_base +"/DPmax_150/"+ region + "_DPmax" + ".tif"
    save_raster(filename_DP, DPmax, haz_tif, im_row, im_col,xmin, xmax, ymin, ymax, raster_image)

    #DA&DP finish
    
    #3. Calculate start of impact: sd_NTL (days start from sd_doy--1)
    check_memory()  


    aff_rows_event, aff_cols_event = np.where((DAmax >= s_t))
    
    sd_NTL = np.zeros((rows_aff, cols_aff))
    sd_NTL[DAmax <= s_t] = 9999 #When DAmax remains equal to or under the threshold, the pixel is not affected 
    
    if len(aff_rows_event)>0:
        for k in range(0,len(aff_rows_event)):
            ii = aff_rows_event[k]
            jj = aff_cols_event[k]  
            d1 = stacked_NTL[:,ii,jj].copy() 
            d1[0:idx_event_start] = np.nan

            #Then we identify days where post-event NTL drops below the threshold (affected days)
            affected_days = np.where(d1<=(pre_event_mean[ii,jj] - s_t * pre_event_std[ii,jj]))  

            sd_NTL[ii,jj] = np.min(affected_days[0]) - BUFFER_pre + 1 #finding first affected day
            #the - buffer +1 makes 1 the start of the event (so sd_NTL=1 means the start of the impact is the same as the start of the event)

    filename_sd = output_base + "/sd_NTL/"+region+ "_sd" + ".tif"
    save_raster(filename_sd, sd_NTL, haz_tif, im_row, im_col,xmin, xmax, ymin, ymax, raster_image)

    #sd_NTL  finish
    
    #4. affected area 0: no, 1: sure affected, 2: predict affect,  DA > s_t is sure affected
    sd_NTL1 = sd_NTL.copy()
    sd_NTL2 = sd_NTL.copy()

    #Identify for certain affected pixels
    aff = np.zeros((rows_aff, cols_aff))
    aff[aff_rows_event, aff_cols_event] = 1   

    #Identify pixels that are indicated affected but very low data availability (<50%) -- so not for sure affected, but predicted
    aff_rows_event1, aff_cols_event1 = np.where((DAmax < s_t) & (avail_perc < 0.5))  #Identify pixels that were not affected but had a very low data availability

    #Loop over affected pixels with low data availability to check if neighbours are affected
    if len(aff_rows_event1) > 0:
        for k in range(0,len(aff_rows_event1)):
            check_memory()  


            ii = aff_rows_event1[k]
            jj = aff_cols_event1[k]
            num_neighb = 0 #number of neighbours
            r = 1 #radius of search neighbour
            
            while (num_neighb == 0) and (r<20):  
                S = sd_NTL1[(ii-r):(ii+r),(jj-r):(jj+r)]
                S[S==9999] = 0
                x = np.nonzero(S)
                neighb = S[x] 
                num_neighb = len(neighb)         
                r += 1 #increase radius
            if (num_neighb > 0):
                sd_n = int(np.min(S[x]))+ BUFFER_pre - 1
                a = stacked_NTL[sd_n,ii,jj]  
                if (a != a):
                    aff[ii,jj] = 2 #predict affected
                    sd_NTL2[ii,jj] = sd_n - BUFFER_pre +1 #adjusted start_date of impact
                    #Again the - buffer +1 makes 1 the start of the event (so sd_NTL=1 means the start of the impact is the same as the start of the event)
    
    #aff shows whether a pixel is affected not, for certain, or predicted (0, 1, 2)
    filename_aff = output_base +"/affected/"+region+ "_aff" +".tif"
    save_raster(filename_aff, aff, haz_tif, im_row, im_col,xmin, xmax, ymin, ymax, raster_image)

    #sd_NTL2 shows the adjusted start date of impact when prediced pixels are included
    filename_sd_adjusted = output_base +"/sd_NTL_neighbours/"+ region + "_sd_neighbours" +".tif"
    save_raster(filename_sd_adjusted, sd_NTL2, haz_tif, im_row, im_col,xmin, xmax, ymin, ymax, raster_image)

    #affected  finish    
    
    #5. recovered  RECOVERY_NOT_FOUND(=999) - not recover within 150days, back to mean +- n * std means recovered
    # List of array names
    # Adjusted is accounting for potential worst impact after event duration but before recovery finished
    # Persistent is accounting for recovery only true when stays recovered
    all_metric_names = [
        "recov_day", "aff_later", "impact_dur", "recov_dur", "worst_day", 
        "recov_day_pers", "aff_later_pers", "impact_dur_pers", "recov_dur_pers", 
    ]

    #normal means worst impact during event duration

    #but worst impact (hence also min NTL, DAmax and DPmax) and recovery time change
    #so:

    #Then persistent means the recovery day changes and thus also impact duration and recovery time. 
    #However, worst impact stays the same (within event duration, like normal)
    #so:
        #worst_day_pers = worst_day 
        #new: recov_day_pers, impact_dur_pers, recov_dur_pers

    #stay the same as normal persistent, but worst impact (and min NTL, DAmax and DPmax) as well as recovery time change

    #Initialise the rasters
    for name in all_metric_names:
        globals()[name] = np.full((rows_aff, cols_aff), NODATA_VALUE)
    
    # NEW: postimpact/postworst availability arrays
    postimpact_avail = np.full((rows_aff, cols_aff), np.nan)
    postworst_avail = np.full((rows_aff, cols_aff), np.nan)

    aff_rows_event2, aff_cols_event2 = np.where((aff > 0)) #find any affected pixels, certainly or predicted 

    #Loop over affected pixels
    if len(aff_rows_event2) > 0:
        
        # Store daily values in CSV
        if store_for_print:
            csv_output_path = os.path.join(output_base, "daily_values", f"{region}_per_pixel_daily.csv")
            os.makedirs(os.path.dirname(csv_output_path), exist_ok=True)
            csv_file = open(csv_output_path, mode="w", newline="")
            writer = csv.writer(csv_file)
            writer.writerow([
                'global_row', 'global_col', 'ntl_i', 'day_i', "DA_i", "recovered", "affected", "aff_later","recovered_pers", "aff_later_pers"
                ])

        for k in range(0, len(aff_rows_event2)):
            check_memory()  


            ii = aff_rows_event2[k]
            jj = aff_cols_event2[k]
            idx_impact_start = int(sd_NTL2[ii,jj]) + BUFFER_pre - 1 #start of the impact, from here onwards we check for recovery
            #the +buffer -1 transforms the sd_NTL (which is 1 for first day of the impact) again to a relative index
            #If the impact starts at the first day of the event then instead of 1 it becomes 1+150+1 = 150 again
            pixel_ts = stacked_NTL[:,ii,jj].copy() #get timeseries of pixel
            
            valid_range = pixel_ts[idx_impact_start:idx_event_end]
            
            if np.all(np.isnan(valid_range)):
                #This exception can occur when the pixel is determined affected based on neighbouring pixels, but has no data itself

                # too little data
                for name in all_metric_names:
                    globals()[name][ii, jj] = NOT_ENOUGH_DATA #Assing insufficient data to all the metrics

            else:              
                mean = pre_event_mean[ii,jj].copy() 
                stdm = pre_event_std[ii,jj].copy() 

                ts_recovered = pixel_ts - (mean - s_t * stdm)                  
                ts_recovered[ts_recovered > 0] = 1 #all positive values are assigned a 1 (=recovered)

                lowest_dip_relative = np.nanargmin(valid_range)  # index within the slice
                lowest_dip = lowest_dip_relative + idx_impact_start  # index in full ts
                worst_day[ii,jj] = lowest_dip - BUFFER_pre + 1
                
                # NEW: postimpact/postworst availability per pixel
                if idx_impact_start < total_days_analysis:
                    postimpact_avail[ii, jj] = (
                        np.sum(~np.isnan(pixel_ts[idx_impact_start:])) /
                        (total_days_analysis - idx_impact_start)
                    )
                if lowest_dip < total_days_analysis:
                    postworst_avail[ii, jj] = (
                        np.sum(~np.isnan(pixel_ts[lowest_dip:])) /
                        (total_days_analysis - lowest_dip)
                    )

                ts_recovered[:lowest_dip] = 0 #recovery 0 before lowest dip

                recovered_indices = np.where ((ts_recovered>0)) #so ts_recovered should be full timeseries but 0 until lowest dip
                #Recovered indices has the lenght of total_days_analysis (=buffer_pre + duration + buffer_post)

                if len(recovered_indices[0]) >0: 
                    first_recovery_idx = np.min(recovered_indices[0])

                    recov_day[ii, jj] = first_recovery_idx - BUFFER_pre + 1
                    impact_dur[ii, jj] = first_recovery_idx - idx_impact_start + 1
                    recov_dur[ii, jj] = first_recovery_idx - lowest_dip + 1

                    #Recov_day is again a specific day, so if recov_day is 153 the recov_day counts from the start of the event (idx 150)
                    #150 should be day 1 again (start of the event) so -buffer +1 gives again actual day.
                    #Not necessary for impact_dur and recov_dur as these are relative, but +1 to make inclusive of start and end day            

                    """This is persistent recovery calculation"""
                    #start looking for persistent recovery after first recovery index has passed
                    recov_idx_pers, persist_checked = find_persistent_recovery(ts_recovered, first_recovery_idx, total_days_analysis, min_ratio=0.75, window=14, min_valid=4)
                    
                    if recov_idx_pers is None:   
                        ts_recovered_pers = np.full_like(ts_recovered, np.nan)   
                                    
                        if persist_checked:
                            recov_day_pers[ii, jj] = RECOVERY_NOT_FOUND
                            impact_dur_pers[ii, jj] = RECOVERY_NOT_FOUND
                            recov_dur_pers[ii, jj]= RECOVERY_NOT_FOUND
                            recov_idx_pers = None
                        else:
                            recov_day_pers[ii, jj]= NOT_ENOUGH_DATA
                            impact_dur_pers[ii, jj]= NOT_ENOUGH_DATA
                            recov_dur_pers[ii, jj]= NOT_ENOUGH_DATA
                            recov_idx_pers = None

                    else:
                        #Create timeseries of 0 and 1
                        ts_recovered_pers = np.zeros_like(ts_recovered)
                        ts_recovered_pers[recov_idx_pers:] = 1

                        recov_day_pers[ii, jj]= recov_idx_pers - BUFFER_pre + 1
                        impact_dur_pers[ii, jj]= recov_idx_pers - idx_impact_start + 1
                        recov_dur_pers[ii, jj]= recov_idx_pers - lowest_dip + 1

                        affected_later_pers = aff_later_pers[ii,jj]

                        if t >= idx_event_start:
                            with np.errstate(divide='ignore', invalid='ignore'):
                                DA_today = (pre_mean - ntl_val) / pre_std #value at time t

                            # Check and clean invalid values
                            if not np.isfinite(DA_today) or abs(DA_today) > 999:
                                DA_today = np.nan
                        else:
                            DA_today = np.nan

                        #t only is the index, so 0 represents 1st day of analysis and the event starting at idx 150
                        #this is then day 0 of the event. We add the 150 and +1 to make 1 the day the event starts.   

                        recovered_pers_val = ts_recovered_pers[t]

                        if store_for_print:
                            writer.writerow([
                                row_idx_global, col_idx_global, ntl_val, t - BUFFER_pre + 1, DA_today, recovered, affected, affected_later, recovered_pers_val, affected_later_pers
                            ])
    
        # NEW: availability-adjusted durations and availability outputs
        save_raster(
            f"{post_base}/postimpact_availability/{ID_number}_postimpact_avail.tif",
            postimpact_avail,
            haz_tif,
            im_row,
            im_col,
            xmin,
            xmax,
            ymin,
            ymax,
            raster_image,
        )

        save_raster(
            f"{post_base}/postworst_availability/{ID_number}_postworst_avail.tif",
            postworst_avail,
            haz_tif,
            im_row,
            im_col,
            xmin,
            xmax,
            ymin,
            ymax,
            raster_image,
        )

        # NEW: save pre-event availability
        save_raster(
            f"{post_base}/preevent_availability/{ID_number}_preevent_avail.tif",
            preevent_avail,
            haz_tif,
            im_row,
            im_col,
            xmin,
            xmax,
            ymin,
            ymax,
            raster_image,
        )

        for avail_th in AVAIL_THRESHOLDS:
            pct_avail = int(avail_th * 100)

            recov_dur_avail = recov_dur.copy()
            impact_dur_avail = impact_dur.copy()
            recov_dur_pers_avail = recov_dur_pers.copy()
            impact_dur_pers_avail = impact_dur_pers.copy()

            avail_mask = postworst_avail < avail_th
            recov_dur_avail[avail_mask] = NOT_ENOUGH_DATA
            impact_dur_avail[avail_mask] = NOT_ENOUGH_DATA
            recov_dur_pers_avail[avail_mask] = NOT_ENOUGH_DATA
            impact_dur_pers_avail[avail_mask] = NOT_ENOUGH_DATA

            valid_avail = np.isfinite(postworst_avail) & (postworst_avail >= avail_th)

            censored = (recov_dur == RECOVERY_NOT_FOUND) & valid_avail
            if np.any(censored):
                max_recov = duration_event + BUFFER_post - worst_day[censored] + 1
                max_impact = duration_event + BUFFER_post - sd_NTL2[censored] + 1
                recov_dur_avail[censored] = max_recov
                impact_dur_avail[censored] = max_impact

            censored_pers = (recov_dur_pers == RECOVERY_NOT_FOUND) & valid_avail
            if np.any(censored_pers):
                max_recov_p = duration_event + BUFFER_post - worst_day[censored_pers] + 1
                max_impact_p = duration_event + BUFFER_post - sd_NTL2[censored_pers] + 1
                recov_dur_pers_avail[censored_pers] = max_recov_p
                impact_dur_pers_avail[censored_pers] = max_impact_p

            save_raster(
                f"{post_base}/recovery_duration/{ID_number}_recov_dur_avail{pct_avail}.tif",
                recov_dur_avail,
                haz_tif,
                im_row,
                im_col,
                xmin,
                xmax,
                ymin,
                ymax,
                raster_image,
            )

            save_raster(
                f"{post_base}/impact_duration/{ID_number}_impact_dur_avail{pct_avail}.tif",
                impact_dur_avail,
                haz_tif,
                im_row,
                im_col,
                xmin,
                xmax,
                ymin,
                ymax,
                raster_image,
            )

            # NEW: save pers availability outputs
            save_raster(
                f"{post_base}/recovery_duration/{ID_number}_recov_dur_pers_avail{pct_avail}.tif",
                recov_dur_pers_avail,
                haz_tif,
                im_row,
                im_col,
                xmin,
                xmax,
                ymin,
                ymax,
                raster_image,
            )

            save_raster(
                f"{post_base}/impact_duration/{ID_number}_impact_dur_pers_avail{pct_avail}.tif",
                impact_dur_pers_avail,
                haz_tif,
                im_row,
                im_col,
                xmin,
                xmax,
                ymin,
                ymax,
                raster_image,
            )



        metric_file_map = {
            "impact_dur": "impact_duration",
            "recov_day": "recovery_day",
            "recov_dur": "recovery_duration",
            "worst_day": "worst_day",
        }

        for varname, folder in metric_file_map.items():
            array = globals()[varname]
            filename = f"{output_base}/{folder}/{region}_{varname}.tif"
            save_raster(filename, array, haz_tif, im_row, im_col, xmin, xmax, ymin, ymax, raster_image)
        
        if np.any(recov_day_pers != NOT_ENOUGH_DATA):
            metric_file_map_pers = {
                    "recov_day_pers": "recovery_day",
                    "impact_dur_pers": "impact_duration", 
                    "recov_dur_pers": "recovery_duration",
                }

            for varname, folder in metric_file_map_pers.items():
                array = globals()[varname]
                filename = f"{output_base}/{folder}/{region}_{varname}.tif"
                save_raster(filename, array, haz_tif, im_row, im_col, xmin, xmax, ymin, ymax, raster_image)
            
            

        else:
            print("No persistent recovery found")

        if store_for_print:
            csv_file.close()
                    
    else:

            print ("no affected pixels")
else:
    print (f"{hazard_tif_path} does not exist")

        
            
            



           