# -*- coding: utf-8 -*-
"""
Created on Tue Oct 14 10:57:34 2025
@author: valer
"""

import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cm as cm
import numpy as np
import pandas as pd
from astral import LocationInfo
from astral.sun import sun
from metpy.calc import mixing_ratio_from_relative_humidity
from metpy.calc import virtual_potential_temperature
from metpy.units import units
plt.rcParams['figure.dpi'] = 300
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12

# file has temperature data for barge (July 21 - 28), heights 0 - 17 km
filepath = r"C:\Users\valer\Documents\WFIP3\barg.assist.tropoe.z01.c1\barg.assist.tropoe.z01.c1.20240720.000005.nc"
data = xr.open_dataset(filepath,decode_times = "true")
filepath2 = r"C:\Users\valer\Documents\WFIP3\barg.assist.tropoe.z01.combined.revised2.nc"
data2 = xr.open_dataset(filepath2,decode_times = "true")

# grab variables from revised file 
P = data2["pressure"].sel(time="2024-07-20 12:00:00") * units.hPa
rh = data2["rh"].sel(time="2024-07-20 12:00:00")
temp = data2["temperature"].sel(time="2024-07-20 12:00:00") * units.degC
theta = data2["theta"].sel(time="2024-07-20 12:00:00")

# calculate virtual potential temp
mixingRatios = mixing_ratio_from_relative_humidity(P,temp,rh)
theta_v = virtual_potential_temperature(P,temp,mixingRatios)

# collect sunrise/sunset info, useful in plots later
location = LocationInfo(latitude=data.VIP_station_lat, longitude=data.VIP_station_lon, timezone="UTC")
date = pd.to_datetime(data.time.values[0])
s = sun(location.observer, date=date)
sunrise = s["sunrise"]
sunset = s["sunset"]

# change the height units on original file
data_1 = data.copy()
data_1 = data_1.assign_coords(height = data_1["height"] * 1000)
data_1["height"].attrs["units"] = "m"

# visualize Nicola's suggestion (interpolate from larger range)
height1 = data_1["height"].sel(height = slice(40,300))
theta1 = data_1["theta"].sel(height = slice(30,350))
theta1 = theta1.sel(time="2024-07-20 12:00:00",method="nearest")
theta1 = theta1.transpose()
# interpolate original theta for reference
interp_theta = theta1.interp(height = np.linspace(40,300,14))
interp_theta = interp_theta.transpose()

# plt.figure(figsize=(10,6))
# interp_theta.plot(y="height",marker='o')
# theta1.plot(y="height",marker='s')
# plt.xlabel('Potential Temperature, θ (K)')
# plt.ylabel('Height (m)')
# plt.xlim(293.8,294.75)
# plt.title(' ')
# plt.legend(['Interpolated ASSIST', 'Original ASSIST'])

# validate Nicola's interpolation with combined file
height2 = data2["height"].sel(height=slice(40,300))
theta.transpose() # already interpolated during combineFiles script

plt.figure(figsize=(10,6))
theta.plot(y="height",marker='o')
theta1.plot(y="height",marker='s')
plt.xlabel('Potential Temperature, θ (K)')
plt.ylabel('Height (m)')
plt.xlim(293.8,294.75)
plt.title(' ')
plt.legend(['Combined ASSIST', 'Original ASSIST'])

# visualize Nicola's suggestion (virtual potential temp)
# interpolate temp, pressure, rh
temp1 = data_1["temperature"].sel(height = slice(30,350))
temp1 = temp1.sel(time="2024-07-20 12:00:00",method="nearest")
temp1 = temp1.transpose()
interp_temp1 = temp1.interp(height = np.linspace(40,300,14)) * units.degC
interp_temp1 = interp_temp1.transpose()
P1 = data_1["pressure"].sel(height=slice(30,350))
P1 = P1.sel(time="2024-07-20 12:00:00",method="nearest")
P1 = P1.transpose()
interp_P1 = P1.interp(height = np.linspace(40,300,14)) * units.hPa
interp_P1 = interp_P1.transpose()
rh1 = data_1["rh"].sel(height=slice(30,350))
rh1 = rh1.sel(time="2024-07-20 12:00:00",method="nearest")
rh1 = rh1.transpose()
interp_rh1 = rh1.interp(height = np.linspace(40,300,14))
interp_rh1 = interp_rh1.transpose()
# cakculate virtual potential temp
mixingRatios1 = mixing_ratio_from_relative_humidity(interp_P1,interp_temp1,interp_rh1)
theta_v1 = virtual_potential_temperature(interp_P1,interp_temp1,mixingRatios1)

theta_v = theta_v.transpose()
theta_v1 = theta_v1.transpose()

plt.figure(figsize=(10,6))
theta_v.plot(y="height",marker='o')
theta_v1.plot(y="height",marker='s')
plt.xlabel('Virtual Potential Temperature, θ_v (K)')
plt.ylabel('Height (m)')
plt.xlim(293.8,296.8)
plt.title(' ')
plt.legend(['Combined ASSIST', 'Original ASSIST'])

# # plotting theta along height and time
# plt.figure(figsize=(10, 5))
# theta.plot(x="time", y="height", cmap="plasma")
# ax = plt.gca()
# ax.xaxis.set_major_formatter(mdates.DateFormatter("%H"))  # only show hours
# ax.axvline(sunrise,color="orange",linestyle="--",linewidth=1.5,label='Sunrise')
# ax.axvline(sunset,color="purple",linestyle="--",linewidth=1.5,label='Sunset')
# ax.legend(loc="upper right")
# plt.title("Potential Temperature vs Height and Time on 20 July, 2024")
# plt.xlabel("UTC Time")
# plt.ylabel("Height (m)")
# plt.tight_layout()
# plt.show()

# # make dTheta into a dataframe
# # first need to take averages of each pair of timesteps
# dTheta = theta.differentiate("height") # central difference dT/dz (2nd order accurate)
# dTheta_20 = dTheta.resample(time="20min",base=0).mean()
# dTheta_dataset = xr.Dataset({"Potential Temperature": theta, "dTheta/dz": dTheta})
# dTheta_dataframe = dTheta_dataset.to_dataframe().reset_index()

# # plotting dTheta along height AND time
# plt.figure(figsize=(10, 5))
# dTheta.plot(x="time", y="height", cmap="plasma")
# ax = plt.gca()
# ax.xaxis.set_major_formatter(mdates.DateFormatter("%H"))  # only show hours
# ax.axvline(sunrise,color="white",linestyle="--",linewidth=1.5,label='Sunrise')
# ax.axvline(sunset,color="black",linestyle="--",linewidth=1.5,label='Sunset')
# ax.legend(loc="upper right")
# plt.title("Change in Potential Temperature vs Height and Time on 15 July, 2024")
# plt.xlabel("UTC Time")
# plt.ylabel("Height (m)")
# plt.tight_layout()
# plt.show()

# # plotting dTheta at specific heights along time
# plt.figure(figsize=(7,7))
# cmap = cm.get_cmap("Blues") # choose my colormap
# height = [46,280]  # only for specific heights: 46, 61, 77, 95, 114, 136, 159, 185, 214, 245, 280
# colors = cmap(np.linspace(0.4,0.8,len(height)))

# # plot for all heights...
# for h, color in zip(height,colors):
#     dTheta_20.sel(height=h).plot.line(x="time",hue="height",label=f"{h} m",color=color)

# ax = plt.gca()
# ax.xaxis.set_major_formatter(mdates.DateFormatter("%H"))  # only show hours
# ax.axvline(sunrise,color="orange",linestyle="--",linewidth=1.5,label='Sunrise')
# ax.axvline(sunset,color="purple",linestyle="--",linewidth=1.5,label='Sunset')
# ax.axhline(y=0,color="black",linestyle="--",linewidth=1.5)
# ax.legend(loc="upper right")
# plt.title("Change in Potential Temperature on 15 July, 2024 between heights")
# plt.xlabel("UTC Time")
# plt.ylabel("dTheta/dz (K/m)")
# plt.tight_layout()
# plt.show()

# # plot vertical resolution for all heights
# all_heights = data_1.height
# plt.hlines(y=all_heights,xmin=0.5,xmax=1.5,colors='r')
# intervals = np.array([100,1000,2000,5000,10000,15000,17000])
# plt.xlim(0,2)
# plt.xticks([])
# plt.yticks(intervals)
# plt.ylabel('Height (m)')
# plt.title('ASSIST Spatial Resolution')
# plt.show()



