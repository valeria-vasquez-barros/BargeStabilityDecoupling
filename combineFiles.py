# -*- coding: utf-8 -*-
"""
Created on Sun Jan 18 15:17:47 2026
@author: valer
"""

import xarray as xr
import numpy as np
import pandas as pd
import os

timesteps = pd.date_range(start="2024-05-21",end="2024-09-30 23:59:59",freq="10T")
heights = np.linspace(40,300,14)

wind_speed = np.full((len(timesteps),len(heights)),np.nan)
wind_direction = np.full((len(timesteps),len(heights)),np.nan)

dataset = xr.Dataset(
    {
        "wind_speed": (["time","height"], wind_speed),
        "wind_direction": (["time","height"], wind_direction),
        "day": ("time", timesteps.floor("D"))
    },
    coords=
    {
         "time": timesteps,
         "height": heights,
    },
)


folder = r"C:\Users\valer\Documents\WFIP3\barg.lidar.z02.a0\downloader"
# folder = r"C:\Users\valer\Documents\WFIP3\lidar.test"

for file in os.listdir(folder):
    # only grab netCDF files
    if not file.endswith(".nc"):
        continue
    fpath = os.path.join(folder, file)
    data = xr.open_dataset(fpath,decode_times = True)
    # grab wind speed/direction from each file (represents one or part of a day)
    ws = data["wind_speed"]
    wd = data["wind_direction"]
    # reindex to match the pre-defined grid of the dataset and ensure coordinate alignment
    ws = ws.reindex_like(dataset["wind_speed"])
    wd = wd.reindex_like(dataset["wind_direction"])
    # put the wind speed/direction data from each day into the dataset
    dataset["wind_speed"]=dataset["wind_speed"].combine_first(ws)
    dataset["wind_direction"]=dataset["wind_direction"].combine_first(wd)
    
print(dataset["wind_speed"].sel(time=slice("2024-07-20","2024-07-20 23:59:59")).values)
print(dataset["wind_direction"].sel(time=slice("2024-07-20","2024-07-20 23:59:59")).values)

new_folder = r"C:\Users\valer\Documents\WFIP3\lidar.test"
new_filename = "barg.lidar.z02.combined.nc"
dataset.to_netcdf(os.path.join(new_folder,new_filename))
print("file saved")
