# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 12:25:13 2026

@author: valer
"""
import xarray as xr
import pandas as pd
import numpy as np

filepathLidar1 = r"C:\Users\valer\Documents\WFIP3\lidar.test\barg.lidar.z02.combined.nc"
dataLidar1 = xr.open_dataset(filepathLidar1,decode_times="true")

filepathLidar2 = r"C:\Users\valer\Documents\WFIP3\wfip3.barg.lidar.z02.b0.nc"
dataLidar2 = xr.open_dataset(filepathLidar2)#,decode_times="true")

# Update the 'time' coordinate in the xarray dataset to the converted datetimes
dataLidar2 = dataLidar2.assign_coords(time=pd.to_datetime(dataLidar2.time.values, unit="s"))
dataLidar2['z'] = xr.DataArray(dataLidar2.Z.values, dims=["z"])
dataLidar2 = dataLidar2.rename({"z": "height"})
# Reorder dimensions so that 'time' is the first dimension
dataLidar2 = dataLidar2.transpose('time', 'height')
dataLidar2["time"]=dataLidar2["time"].dt.ceil("10min")

# Specify days ON-station, excludes off-station
dates1 = pd.date_range(start="2024-06-17 05:00:00",end="2024-06-23 11:00:00",freq="10T")
dates2 = pd.date_range(start="2024-06-29 05:00:00",end="2024-08-08 11:00:00",freq="10T")
dates3 = pd.date_range(start="2024-08-23 06:00:00",end="2024-09-28 10:00:00",freq="10T")
valid = dates1.union(dates2).union(dates3)

onStationA = dataLidar1.time.isin(valid).sel(time=slice("2024-06-18 00:00:00", "2024-09-19 23:50:00"))
onStationL = dataLidar2.time.isin(valid).sel(time=slice("2024-06-18 00:00:00", "2024-09-19 23:50:00"))

dataLidar1 = dataLidar1.where(onStationA)
dataLidar2 = dataLidar2.where(onStationL)

# Specify times that data is available, excludes unavailable data among all instruments
lidar1Avail_s = dataLidar1["wind_direction"].sel(time=slice("2024-06-18","2024-09-19"),height=slice(40,60)).notnull().all("height")
lidar1Avail_h = dataLidar1["wind_direction"].sel(time=slice("2024-06-18","2024-09-19"),height=slice(120,160)).notnull().all("height")
lidar2Avail_s = dataLidar2["WD"].sel(time=slice("2024-06-18","2024-09-19"),height=slice(40,60)).notnull().all("height")
lidar2Avail_h = dataLidar2["WD"].sel(time=slice("2024-06-18","2024-09-19"),height=slice(120,160)).notnull().all("height")

overlap = (lidar1Avail_s & lidar1Avail_h & lidar2Avail_s & lidar2Avail_h)

dataLidar1 = dataLidar1.where(overlap)
dataLidar2 = dataLidar2.where(overlap)