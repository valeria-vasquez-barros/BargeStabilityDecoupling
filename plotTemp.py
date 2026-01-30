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

# file has temperature data for barge (July 21 - 28), heights 0 - 17 km
filepath = r"C:\Users\valer\Documents\WFIP3\barg.assist.tropoe.z01.c1\barg.assist.tropoe.z01.c1.20240720.000005.nc"
data = xr.open_dataset(filepath,decode_times = "true")

# collect sunrise/sunset info, useful in plots later
location = LocationInfo(latitude=data.VIP_station_lat, longitude=data.VIP_station_lon, timezone="UTC")
date = pd.to_datetime(data.time.values[0])
s=sun(location.observer, date=date)
sunrise = s["sunrise"]
sunset = s["sunset"]

# copy the data to change the height units
data_1 = data.copy()
data_1 = data_1.assign_coords(height = data_1["height"] * 1000)
data_1["height"].attrs["units"] = "m"

# only looking at heights 40-300 m
theta = data_1["theta"].sel(height = slice(40,300))
height = data_1["height"].sel(height = slice(40,300))

# plotting theta along height and time
plt.figure(figsize=(10, 5))
theta.plot(x="time", y="height", cmap="plasma")
ax = plt.gca()
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H"))  # only show hours
ax.axvline(sunrise,color="orange",linestyle="--",linewidth=1.5,label='Sunrise')
ax.axvline(sunset,color="purple",linestyle="--",linewidth=1.5,label='Sunset')
ax.legend(loc="upper right")
plt.title("Potential Temperature vs Height and Time on 20 July, 2024")
plt.xlabel("UTC Time")
plt.ylabel("Height (m)")
plt.tight_layout()
plt.show()

# make dTheta into a dataframe
# first need to take averages of each pair of timesteps
dTheta = theta.differentiate("height") # central difference dT/dz (2nd order accurate)
dTheta_20 = dTheta.resample(time="20min",base=0).mean()
dTheta_dataset = xr.Dataset({"Potential Temperature": theta, "dTheta/dz": dTheta})
dTheta_dataframe = dTheta_dataset.to_dataframe().reset_index()

# plotting dTheta along height AND time
plt.figure(figsize=(10, 5))
dTheta.plot(x="time", y="height", cmap="plasma")
ax = plt.gca()
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H"))  # only show hours
ax.axvline(sunrise,color="white",linestyle="--",linewidth=1.5,label='Sunrise')
ax.axvline(sunset,color="black",linestyle="--",linewidth=1.5,label='Sunset')
ax.legend(loc="upper right")
plt.title("Change in Potential Temperature vs Height and Time on 15 July, 2024")
plt.xlabel("UTC Time")
plt.ylabel("Height (m)")
plt.tight_layout()
plt.show()

# plotting dTheta at specific heights along time
plt.figure(figsize=(7,7))
cmap = cm.get_cmap("Blues") # choose my colormap
height = [46,280]  # only for specific heights: 46, 61, 77, 95, 114, 136, 159, 185, 214, 245, 280
colors = cmap(np.linspace(0.4,0.8,len(height)))

# plot for all heights...
for h, color in zip(height,colors):
    dTheta_20.sel(height=h).plot.line(x="time",hue="height",label=f"{h} m",color=color)

ax = plt.gca()
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H"))  # only show hours
ax.axvline(sunrise,color="orange",linestyle="--",linewidth=1.5,label='Sunrise')
ax.axvline(sunset,color="purple",linestyle="--",linewidth=1.5,label='Sunset')
ax.axhline(y=0,color="black",linestyle="--",linewidth=1.5)
ax.legend(loc="upper right")
plt.title("Change in Potential Temperature on 15 July, 2024 between heights")
plt.xlabel("UTC Time")
plt.ylabel("dTheta/dz (K/m)")
plt.tight_layout()
plt.show()



