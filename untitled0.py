# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 09:39:48 2026

@author: valer
"""

# make a smaller combineFiles script with just two files to troubleshoot

import xarray as xr
import numpy as np
import pandas as pd
import os

timesteps = pd.date_range(start="2024-07-15 00:00:00",
                          end="2024-07-20 23:59:59",
                          freq="10T")
heights = np.linspace(40,300,14)

# new folder with just two copied files (2024-07-15 and 2024-07-20)
folder = r"C:\Users\valer\Documents\WFIP3\assist.test"

files = [
    os.path.join(folder,file)
    for file in os.listdir(folder)
    if file.endswith(".nc")
]

def preprocess(file_data):
    # ensure correct timestamp
    # print("Timestamps BEFORE processing:",file_data["time"]) # passes
    file_data["time"]=file_data["time"].dt.floor("10min")
    # print("Timesteps AFTER processing:",file_data["time"]) # passes
    # km to m
    # print(file_data["height"]) # passes
    file_data=file_data.assign_coords(height=file_data.height*1000)
    file_data["height"].attrs["units"] = "m"
    # print(file_data["height"]) # passes
    # pre-select the desired height range
    file_data=file_data.sel(height=slice(40,300))
    file_data=file_data.interp(height = heights,kwargs={"fill_value":"extrapolate"})
    # print("theta coordinates:",file_data["theta"]) # passes
    # print("temp coordinates:",file_data["temperature"]) # passes
    return file_data[["theta","temperature","time"]]

data_comb = xr.open_mfdataset(
    files, 
    combine="nested", 
    concat_dim="time",
    preprocess=preprocess,
    coords="minimal",
    compat="override"
    )

data_comb = data_comb.sortby("time")
data_comb = data_comb.reindex(time=timesteps)
new_folder = r"C:\Users\valer\Documents\WFIP3\assist.test"
new_filename = "assist.combined.test.nc"
data_comb.to_netcdf(os.path.join(new_folder,new_filename))
print("file saved")
