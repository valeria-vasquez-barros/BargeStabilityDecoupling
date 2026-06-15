# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 12:25:13 2026

@author: valer
"""
import xarray as xr

filepathLidar = r"C:\Users\valer\Documents\WFIP3\wfip3.barg.lidar.z05.c0.nc"
dataLidar = xr.open_dataset(filepathLidar,decode_times="true")

wind_speed = dataLidar["WS"]
