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
theta = data_0["theta"].sel(height = slice(40,300), time = slice("2024-07-20 00:00:00","2024-07-20 23:50:50"))
temp = data_0["temperature"].sel(height = slice(40,300),time = slice("2024-07-20 00:00:00","2024-07-20 23:50:50"))
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

ws1 = data1["wind_speed"].sel(time = slice("2024-07-20 00:00:00","2024-07-20 23:50:50"))
ws2 = data2["wind_speed"].sel(time = slice("2024-07-20 00:00:00","2024-07-20 23:50:50"))
wind_speed = xr.concat([ws1,ws2],dim="time").sortby("time")

wd1 = np.deg2rad(data1["wind_direction"]).sel(time = slice("2024-07-20 00:00:00","2024-07-20 23:50:50"))
wd2 = np.deg2rad(data2["wind_direction"]).sel(time = slice("2024-07-20 00:00:00","2024-07-20 23:50:50"))
wind_direction = xr.concat([wd1,wd2],dim="time").sortby("time")

uGeo = -wind_speed * np.sin(wind_direction)
vGeo = -wind_speed * np.cos(wind_direction)
uGeo = uGeo.interp(time=thetaExt.time)
vGeo = vGeo.interp(time=thetaExt.time)


g = 9.81
h_i = 40
h_f = 60

dZ = h_f-h_i

# 2) change in potential temperature
theta_i = thetaExt.sel(height=h_i)
theta_f = thetaExt.sel(height=h_f)
deltaTheta = theta_f - theta_i # K

# 3) change in temperature
temp_i = tempExt.sel(height=h_i)
temp_f = tempExt.sel(height=h_f)
dTemp = temp_f - temp_i
# avgTemp_hub = dTemp_hub/dZ_hub + 273.15 # K, apparently this could be a lapse rate?
avgTemp = (temp_i+temp_f)/2 # K

# 4) change in u,v over heights
u_i = uGeo.sel(height=h_i)
u_f = uGeo.sel(height=h_f)
dU = u_f - u_i
v_i = vGeo.sel(height=h_i)
v_f = vGeo.sel(height=h_f)
dV = v_f - v_i

# 5) final calculation
num1 = g/avgTemp
num2 = deltaTheta*dZ
sGeo = (dU**2+dV**2)
num3 = num2/sGeo
BulkRi = num1*num3

fig, ax = plt.subplots(figsize=(8,4))
BulkRi.plot(ax=ax)

ax.set_xlim(BulkRi.time.min().values,BulkRi.time.max().values)
ax.set_ylim(-1, 1)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H"))
ax.axhline(0.25, linestyle="--", label='Critical Ri')
# ax.axvline(sunrise, linestyle="--", label='Sunrise')
# ax.axvline(sunset, linestyle="--", label='Sunset')

ax.set_title("20 July 2024 (40-60m)")
ax.set_xlabel("UTC Time")
ax.set_ylabel("Bulk Richardson number")
ax.legend()
plt.tight_layout()
plt.show()
