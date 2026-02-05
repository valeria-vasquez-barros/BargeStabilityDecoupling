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

theta = np.full((len(timesteps),len(heights)),np.nan)
temperature = np.full((len(timesteps),len(heights)),np.nan)

dataset = xr.Dataset(
    {
        "theta": (["time","height"], theta),
        "temperature": (["time","height"], temperature),
        "day": ("time", timesteps.floor("D"))
    },
    coords=
    {
         "time": timesteps,
         "height": heights,
    },
)

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
    # pre-select the desired height range
    file_data["theta"]=file_data["theta"].sel(height=slice(40,300))
    file_data["theta"]=file_data["theta"].interp(height = heights,
                                                 kwargs={"fill_value":"extrapolate"}
                                                 )
    file_data["temperature"]=file_data["temperature"].sel(height=slice(40,300))
    file_data["temperature"]=file_data["temperature"].interp(height = heights,
                                                 kwargs={"fill_value":"extrapolate"}
                                                 )
    return file_data[["theta","temperature"]]

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
# data_comb = data_comb.assign_coords(day=("time",timesteps.floor("D")))
dataset["theta"] = data_comb["theta"]
dataset["temperature"] = data_comb["temperature"]

    # data = xr.open_dataset(fpath,decode_times = True)
    # data = data.reindex_like(time=dataset)
    # dataset["theta"].loc[dict(time=data.time,height=data.height)] = data["theta"]
    # dataset["temperature"].loc[dict(time=data.time,height=data.height)] = data["temperature"]
    # grab variables from each file (represents one or part of a day)
    # theta = data["theta"]
    # temperature = data["temperature"]
    # # reindex to match the pre-defined grid of the dataset and ensure coordinate alignment
    # theta = theta.reindex_like(dataset["theta"])
    # temperature = temperature.reindex_like(dataset["temperature"])
    # # put the data from each day into the dataset
    # dataset["theta"]=dataset["theta"].combine_first(theta)
    # dataset["temperature"]=dataset["temperature"].combine_first(temperature)
    
# print(dataset["theta"].sel(time=slice("2024-07-20","2024-07-20 23:59:59")).values)
# print(dataset["temperature"].sel(time=slice("2024-07-20","2024-07-20 23:59:59")).values)

new_folder = r"C:\Users\valer\Documents\WFIP3"
new_filename = "barg.assist.tropoe.z01.combined.nc"
dataset.to_netcdf(os.path.join(new_folder,new_filename))
print("file saved")
