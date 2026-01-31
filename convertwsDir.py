# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 22:13:22 2025

@author: valer
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import xarray as xr

# file has wind speed data for barge (July 21- 28), heights 40 - 300 m, deltaH = 20 m
filepath1 = r"C:\Users\valer\Documents\WFIP3\barg.lidar.z02.a0\downloader\barg.lidar.z02.a0.20240715.001000.sta.nc"
data1 = xr.open_dataset(filepath1,decode_times = "true")
# filepath2 = r"C:\Users\valer\Documents\WFIP3\barg.lidar.z02.a0\downloader\barg.lidar.z02.a0.20240720.120000.sta.nc"
# data2 = xr.open_dataset(filepath2,decode_times = "true")

# manipulate variables
ws1 = data1["wind_speed"]
# ws2 = data2["wind_speed"]
# wind_speed = xr.concat([ws1,ws2],dim="time").sortby("time")
wind_speed = ws1

wd1 = np.deg2rad(data1["wind_direction"])
# wd2 = np.deg2rad(data2["wind_direction"])
# wind_direction = xr.concat([wd1,wd2],dim="time").sortby("time")
wind_direction = wd1

uGeo = -wind_speed * np.sin(wind_direction)
vGeo = -wind_speed * np.cos(wind_direction)
sGeo = np.sqrt(uGeo**2+vGeo**2)

# plot wind speed
plt.figure(figsize=(10, 5))
sGeo.plot(x="time", y="height", cmap="YlGnBu", cbar_kwargs={"label": "wind speed (m/s)"})
ax = plt.gca()
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H"))  # only show hours
plt.title("Barge wind speed on 20 July, 2024")
plt.xlabel("Time")
plt.ylabel("Height (m)")
plt.tight_layout()
plt.show()

# plot wind direction
fig,ax = plt.subplots(figsize=(10,5))
T,Z = np.meshgrid(uGeo["time"],uGeo["height"],indexing="ij")
q = ax.quiver(T[::5],Z[::5],uGeo[::5],vGeo[::5],sGeo[::5])
plt.colorbar(q,ax=ax,label="Wind Speed (m/s)")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H"))  # only show hours
ax.set_xlabel("UTC Time")
ax.set_ylabel("Height (m)")
ax.set_title("Wind Vector Field on 15 July, 2024")
plt.show()

# make a dataframe of u, v, s, du, and dv
du = uGeo.differentiate("height")
dv = vGeo.differentiate("height")
du_20 = du.resample(time="20min",base=0).mean()
dv_20 = dv.resample(time="20min",base=0).mean()
uv_dataset = xr.Dataset({"uGeo": uGeo, "vGeo": vGeo, "sGeo": sGeo, "du/dz": du_20, "dv/dz": dv_20})
uv_dataframe = uv_dataset.to_dataframe().reset_index()
print(uv_dataframe)

