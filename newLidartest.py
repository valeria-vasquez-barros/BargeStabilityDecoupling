# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 12:25:13 2026

@author: valer
"""
import xarray as xr

filepathLidar = r"C:\Users\valer\Documents\WFIP3\wfip3.barg.lidar.z05.c0.nc"
dataLidar = xr.open_dataset(filepathLidar)#,decode_times="true")

# Update the 'time' coordinate in the xarray dataset to the converted datetimes
dataLidar['time'] = xr.DataArray(dataLidar.base_time.values, dims=["time"])
dataLidar['z'] = xr.DataArray(dataLidar.Z.values, dims=["z"])
dataLidar = dataLidar.rename({"z": "height"})
# Reorder dimensions so that 'time' is the first dimension
dataLidar = dataLidar.transpose('time', 'height')

