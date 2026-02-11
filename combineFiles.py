# -*- coding: utf-8 -*-
"""
Created on Sun Jan 18 15:17:47 2026
@author: valer
"""

import xarray as xr
import numpy as np
import pandas as pd
import os

# adjust based on data files
timesteps = pd.date_range(start="2024-05-24 00:00:00",
                          end="2024-09-19 23:59:59",
                          freq="10T")
heights = np.linspace(40,300,14)

folder = r"C:\Users\valer\Documents\WFIP3\barg.assist.tropoe.z01.c1"

files = [
    os.path.join(folder,file)
    for file in os.listdir(folder)
    if file.endswith(".nc")
]

def preprocess(file_data):
    # ensure correct timestamp
    file_data["time"]=file_data["time"].dt.floor("10min")
    # km to m
    file_data=file_data.assign_coords(height=file_data.height*1000)
    file_data["height"].attrs["units"] = "m"
    # pre-select the desired height range
    file_data=file_data.sel(height=slice(40,300))
    file_data=file_data.interp(height = heights,kwargs={"fill_value":"extrapolate"})
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

new_folder = r"C:\Users\valer\Documents\WFIP3"
new_filename = "barg.assist.tropoe.z01.combined.nc"
data_comb.to_netcdf(os.path.join(new_folder,new_filename))
print("file saved")
