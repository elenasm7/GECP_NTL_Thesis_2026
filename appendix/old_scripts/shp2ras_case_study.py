#!/usr/bin/env python
# coding=utf-8

#Only necessary for multi, avoid_single and avoid_multi. Single has been calcualted by Yang

import matplotlib
matplotlib.use('Agg')
from osgeo import gdal
from osgeo import ogr  
import shapefile
ogr.UseExceptions()
import os
import random
import sys

gdal.PushErrorHandler('CPLQuietErrorHandler')

recalc = False
region = "philippines"

folder_base = "data"
polygon_folder = f"{folder_base}/shape_files/buffered_files/"
# ntlthesis/data/shape_files/buffered_files

# /PH_dissolved_buffer.shp
i=0

for filename in os.listdir(polygon_folder):
    print (filename)
    if filename.endswith(".shp"):
                        
        row=2400
        col=2400
        
        shp_file = f"{polygon_folder}/PH_dissolved_buffer.shp"

        ds_shp=shapefile.Reader(shp_file)  

        shapes = ds_shp.shapes()
        if not shapes:
            print(f"Shapefile has no shapes. Skipping.")
            continue

        try:
            minX, minY, maxX, maxY = shapes[0].bbox 
        except AttributeError:
            points = shapes[0].points
            if not points:
                continue
            xs, ys = zip(*points)
            minX, maxX = min(xs), max(xs)
            minY, maxY = min(ys), max(ys)
        
        h_min = int((minX + 180)//10)
        h_max = int((maxX + 180)//10)
        v_min = int((90 - maxY)//10)
        v_max = int((90 - minY)//10)
        
        # root_dir_NTL = "/gpfs/work4/0/FWC2/MYRIAD/data/NTL_data/Processed_data/Black_Marble_processed/corrected_NTL/2013/h00v01/" #contains NTL data
            
        # tifs = [f for f in os.listdir(root_dir_NTL)]
        # random_file = random.choice(tifs)
        # NTL_path = os.path.join(root_dir_NTL, random_file)
        # NTL_ds = gdal.Open(NTL_path)    
        im_proj = "EPSG:4326"

        #shp2raster
        raster_folder = f"{folder_base}/vectors/"
    
        output_path = os.path.join(raster_folder, f"{region}.tif")
        if os.path.exists(output_path) and not recalc:
            print (output_path)
            print ("already calculated")
            continue

        if not ((h_max < 0 or h_min > 35) or (v_max < 0 or v_min > 35)):

            h_min = max(0, min(h_min, 35))
            h_max = max(0, min(h_max, 35))
            v_min = max(0, min(v_min, 17))
            v_max = max(0, min(v_max, 17))
            deltah = h_max - h_min + 1
            deltav = v_max - v_min + 1
        
            h = h_min
            v = v_min

            left = -180 + h * 10
            top = 90 - v * 10

            im_row = int(deltav * row)
            im_col = int(deltah * col)
        
            res = 0.004166666666666673
            im_geotrans = [left, res, 0, top, 0, -res]

            if not(os.path.exists(raster_folder)):
                os.makedirs(raster_folder)

            ds_shp = ogr.Open(shp_file)
            shp_layer = ds_shp.GetLayer()

            target_ds = gdal.GetDriverByName('GTiff').Create(raster_folder+region+'.tif', xsize=im_col, ysize=im_row, bands=1, eType=gdal.GDT_Byte)
            target_ds.SetGeoTransform(im_geotrans)
            target_ds.SetProjection(im_proj)
            band = target_ds.GetRasterBand(1)
            band.SetNoDataValue(-9999)
            band.FlushCache()
            gdal.RasterizeLayer(target_ds,[1],shp_layer)                        
        else:
            print("out of boundary")


        
                