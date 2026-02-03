# -*- coding: utf-8 -*-
"""
Created on Tue Jan 6 20:04:17 2026

@author: valer
"""

import xarray as xr
import numpy as np
import pandas as pd
from astral import LocationInfo
from astral.sun import sun
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Open the data files
filepath0 = r"C:\Users\valer\Documents\WFIP3\barg.assist.tropoe.z01.c1\barg.assist.tropoe.z01.c1.20240720.000005.nc"
filepathTest = r"C:\Users\valer\Documents\WFIP3\lidar.test\barg.lidar.z02.combined.nc"
filepath1 = r"C:\Users\valer\Documents\WFIP3\barg.lidar.z02.a0\downloader\barg.lidar.z02.a0.20240720.001000.sta.nc"
filepath2 = r"C:\Users\valer\Documents\WFIP3\barg.lidar.z02.a0\downloader\barg.lidar.z02.a0.20240720.120000.sta.nc"
data0 = xr.open_dataset(filepath0,decode_times = "true")
dataTest = xr.open_dataset(filepathTest,decode_times = "true")
data1 = xr.open_dataset(filepath1,decode_times="true")
data2 = xr.open_dataset(filepath2,decode_times = "true")

# grab height, theta, temp variables from assist
# note: using 40-300m to match lidar grid size
# data_0 = data0.copy()
# data_0 = data_0.assign_coords(height = data_0["height"] * 1000)
# data_0["height"].attrs["units"] = "m"
# height = data_0["height"].sel(height = slice(40,300))
# theta = data_0["theta"].sel(height = slice(40,300))
# temp = data_0["temperature"].sel(height = slice(40,300))

# # extrapolate temp data
# thetaExt = theta.interp(height = np.linspace(40,300,14),kwargs={"fill_value":"extrapolate"})
# tempExt = temp.interp(height = np.linspace(40,300,14),kwargs={"fill_value":"extrapolate"})
# dTheta = thetaExt.differentiate("height") # central difference dT/dz (2nd order accurate)

# combine wind speeds
wind_speedCombined = dataTest["wind_speed"].sel(time=slice("2024-07-20 00:10:00", "2024-07-21 00:00:00"))
ws1_initial = data1["wind_speed"]
timesLeft = pd.date_range(start='2024-07-20 00:10:00', end='2024-07-20 12:00:00',freq='10T')
ws1 = ws1_initial.reindex(time=timesLeft)
ws2 = data2["wind_speed"]
wind_speedConcat = xr.concat([ws1,ws2],dim="time").sortby("time")

# combine wind directions
wind_directionCombined = np.deg2rad(dataTest["wind_direction"]).sel(time=slice("2024-07-20 00:10:00", "2024-07-21 00:00:00"))
wd1_initial = np.deg2rad(data1["wind_direction"])
wd1 = wd1_initial.reindex(time=timesLeft)
wd2 = np.deg2rad(data2["wind_direction"])
wind_directionConcat = xr.concat([wd1,wd2],dim="time").sortby("time")

# troubleshooting
# print("Combined wind speed/direction time identical:", wind_speedCombined.time.identical(wind_directionCombined.time))
# print("Combined wind speed/direction height identical:", wind_speedCombined.height.identical(wind_directionCombined.height))
# print("Original wind speed/direction time identical:", ws1_initial.time.identical(wd1_initial.time))
# print("Original wind speed/direction height identical:", ws1_initial.height.identical(wd1_initial.height))
print("Combined/original wind speed time identical:", wind_speedCombined.time.identical(wind_speedConcat.time))
print("Combined/original wind speed height identical:", wind_speedCombined.height.identical(wind_speedConcat.height))
print("Combined/original wind direction time identical:", wind_directionCombined.time.identical(wind_directionConcat.time))
print("Combined/original wind direction height identical:", wind_directionCombined.height.identical(wind_directionConcat.height))

# calculate u and v
uGeo = -wind_speedCombined * np.sin(wind_directionCombined)
vGeo = -wind_speedCombined * np.cos(wind_directionCombined)
sGeo = np.sqrt(uGeo**2+vGeo**2)

uGeoConcat = -wind_speedConcat * np.sin(wind_directionConcat)
vGeoConcat = -wind_speedConcat * np.cos(wind_directionConcat)
sGeoConcat = np.sqrt(uGeoConcat**2+vGeoConcat**2)

# plot dTheta along height and time:
# 1) collect sunrise/sunset info
# location = LocationInfo(latitude=data0.VIP_station_lat, longitude=data0.VIP_station_lon, timezone="UTC")
# date = pd.to_datetime(data0.time.values[0])
# s=sun(location.observer, date=date)
# sunrise = s["sunrise"]
# sunset = s["sunset"]

# # 2) contour plot
# plt.figure(figsize=(10, 5))
# dTheta.plot(x="time", y="height", cmap="coolwarm")
# ax = plt.gca()
# ax.xaxis.set_major_formatter(mdates.DateFormatter("%H"))  # only show hours
# ax.axvline(sunrise,color="purple",linestyle="--",linewidth=1.5,label='Sunrise')
# ax.axvline(sunset,color="black",linestyle="--",linewidth=1.5,label='Sunset')
# ax.legend(loc="upper right")
# plt.title("Change in Potential Temperature on 20 July, 2024")
# plt.xlabel("UTC Time")
# plt.ylabel("Height (m)")
# plt.tight_layout()
# plt.show()

# plot wind speed along height and time
fig,ax = plt.subplots(figsize=(10,5))
T,Z = np.meshgrid(uGeo["time"],uGeo["height"],indexing="ij")
q = ax.quiver(T[::5],Z[::5],uGeo[::5],vGeo[::5],sGeo[::5])
plt.colorbar(q,ax=ax,label="Wind Speed (m/s)")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H"))  # only show hours
ax.set_xlabel("UTC Time")
ax.set_ylabel("Height (m)")
ax.set_title("Wind Vector Field [combined lidar file] on 20 July, 2024")
plt.show()

fig,ax = plt.subplots(figsize=(10,5))
T,Z = np.meshgrid(uGeoConcat["time"],uGeoConcat["height"],indexing="ij")
q = ax.quiver(T[::5],Z[::5],uGeoConcat[::5],vGeoConcat[::5],sGeoConcat[::5])
plt.colorbar(q,ax=ax,label="Wind Speed (m/s)")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H"))  # only show hours
ax.set_xlabel("UTC Time")
ax.set_ylabel("Height (m)")
ax.set_title("Wind Vector Field [original lidar file] on 20 July, 2024")
plt.show()

# # calculate Bulk Richardson number:
# # 1) establish height difference
# g = 9.81
# h_i = 60
# h_f = 200
# deltaZ = h_f-h_i

# # 2) change in potential temperature
# theta_i = thetaExt.sel(height=h_i)
# theta_f = thetaExt.sel(height=h_f)
# deltaTheta = theta_f.values - theta_i.values # K

# # 3) change in temperature
# temp_i = tempExt.sel(height=h_i)
# temp_f = tempExt.sel(height=h_f)
# deltaTemp = temp_f.values - temp_i.values
# avgTemp = deltaTemp/deltaZ + 273.15 # K

# # 4) change in u,v over heights
# u_i = uGeo.sel(height=h_i)
# u_f = uGeo.sel(height=h_f)
# deltaU = u_f.values - u_i.values
# v_i = vGeo.sel(height=h_i)
# v_f = vGeo.sel(height=h_f)
# deltaV = v_f.values - v_i.values

# # 5) final calculation
# numerator1 = g/avgTemp
# numerator2 = deltaTheta*deltaZ
# sGeo = (deltaU**2+deltaV**2)
# numerator3 = numerator2/sGeo
# bulk_Ri = numerator1*numerator3
# BulkRi = xr.DataArray(bulk_Ri, coords = {"time": data0.time}, dims = ("time"))

# # plot Bulk Richardson number over time
# BulkRi.plot()
# ax = plt.gca()
# ax.xaxis.set_major_formatter(mdates.DateFormatter("%H"))  # only show hours
# ax.axhline(0.25, color="orange", linestyle="--", label='Critical Ri')
# ax.axvline(sunrise,color="purple",linestyle="--",linewidth=1.5,label='Sunrise')
# ax.axvline(sunset,color="black",linestyle="--",linewidth=1.5,label='Sunset')
# ax.legend(loc="lower right")
# plt.title("[combined lidar file] 20 July, 2024")
# plt.xlabel("UTC Time")
# plt.ylabel("Bulk Richardson number between 60-200 m")
# plt.tight_layout()
# plt.show()

data0.close()
dataTest.close()