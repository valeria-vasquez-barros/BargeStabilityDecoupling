# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 08:28:37 2025

@author: valer
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# assist temp data
filepath0 = r"C:\Users\valer\Documents\WFIP3\barg.assist.tropoe.z01.c1\barg.assist.tropoe.z01.c1.20240720.000005.nc"
data0 = xr.open_dataset(filepath0,decode_times = "true")
data_0 = data0.copy()
data_0 = data_0.assign_coords(height = data_0["height"] * 1000)
data_0["height"].attrs["units"] = "m"
theta = data_0["theta"].sel(height = slice(40,300))
temp = data_0["temperature"].sel(height = slice(40,300))
height = data_0["height"].sel(height = slice(40,300))

# extrapolate temp data
thetaExt = theta.interp(height = np.linspace(40,300,14),kwargs={"fill_value":"extrapolate"})
tempExt = temp.interp(height = np.linspace(40,300,14),kwargs={"fill_value":"extrapolate"})
dTheta = thetaExt.differentiate("height") # central difference dT/dz (2nd order accurate)

# plot for verification
plt.figure(figsize=(10, 5))
dTheta.plot(x="time", y="height", cmap="coolwarm")
ax = plt.gca()
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H"))  # only show hours
plt.title("Change in Potential Temperature vs Height and Time on 20 July, 2024")
plt.xlabel("UTC Time")
plt.ylabel("Height (m)")
plt.tight_layout()
plt.show()

filepath1 = r"C:\Users\valer\Documents\WFIP3\barg.lidar.z02.a0\downloader\barg.lidar.z02.a0.20240720.001000.sta.nc"
data1 = xr.open_dataset(filepath1,decode_times = "true")
filepath2 = r"C:\Users\valer\Documents\WFIP3\barg.lidar.z02.a0\downloader\barg.lidar.z02.a0.20240720.120000.sta.nc"
data2 = xr.open_dataset(filepath2,decode_times = "true")

ws1 = data1["wind_speed"]
ws2 = data2["wind_speed"]
wind_speed = xr.concat([ws1,ws2],dim="time").sortby("time")

wd1 = np.deg2rad(data1["wind_direction"])
wd2 = np.deg2rad(data2["wind_direction"])
wind_direction = xr.concat([wd1,wd2],dim="time").sortby("time")

uGeo = -wind_speed * np.sin(wind_direction)
vGeo = -wind_speed * np.cos(wind_direction)

g = 9.81
h_i = 40
h_f = 60

deltaZ = h_f-h_i

theta_i = thetaExt.sel(height=h_i)
theta_f = thetaExt.sel(height=h_f)
deltaTheta = theta_f.values - theta_i.values # K

temp_i = tempExt.sel(height=h_i)
temp_f = tempExt.sel(height=h_f)
deltaTemp = temp_f.values - temp_i.values
avgTemp = deltaTemp/deltaZ + 273.15 # K

u_i = uGeo.sel(height=h_i)
u_f = uGeo.sel(height=h_f)
deltaU = u_f.values - u_i.values

v_i = vGeo.sel(height=h_i)
v_f = vGeo.sel(height=h_f)
deltaV = v_f.values - v_i.values

numerator1 = g/avgTemp
numerator2 = deltaTheta*deltaZ
sGeo = (deltaU**2+deltaV**2)
numerator3 = numerator2/sGeo
BulkRi = numerator1*numerator3
print(BulkRi.values)
