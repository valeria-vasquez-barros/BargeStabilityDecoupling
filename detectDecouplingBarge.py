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
filepathAssist = r"C:\Users\valer\Documents\WFIP3\barg.assist.tropoe.z01.combined.nc"
filepathLidar = r"C:\Users\valer\Documents\WFIP3\lidar.test\barg.lidar.z02.combined.nc"
dataAssist = xr.open_dataset(filepathAssist,decode_times = "true")
dataLidar = xr.open_dataset(filepathLidar,decode_times="true")

# Specify days on station
dates1 = pd.date_range(start="2024-06-17 05:00:00",end="2024-06-23 11:00:00",freq="10T")
dates2 = pd.date_range(start="2024-06-29 05:00:00",end="2024-08-08 11:00:00",freq="10T")
dates3 = pd.date_range(start="2024-08-23 06:00:00",end="2024-09-28 10:00:00",freq="10T")
valid = dates1.union(dates2).union(dates3)

onStationA = dataAssist.time.isin(valid)
onStationL = dataLidar.time.isin(valid)

dataAssist = dataAssist.where(onStationA)
dataLidar = dataLidar.where(onStationL)

# collect sunrise/sunset info
location = LocationInfo(latitude=dataAssist.VIP_station_lat, longitude=dataAssist.VIP_station_lon, timezone="UTC")
date = pd.to_datetime(dataAssist.time.values[0]) # this is why my location is off?
s=sun(location.observer, date=date)
sunrise = s["sunrise"]
sunset = s["sunset"]

# Grab theta, temp variables from combined assist file
theta = dataAssist["theta"]
temp = dataAssist["temperature"]
tempK = temp + 273.15 # convert to K
dTheta = theta.differentiate("height")

# Select dTheta "near-surface" and "hub-height"
dTheta_surf = dTheta.sel(height=slice(40,60))
dTheta_hub = dTheta.sel(height=slice(120,160))
dTheta_times = dTheta.time

# Static stability quadrant analysis:
dTheta_surf_mean = dTheta_surf.mean("height")
dTheta_hub_mean = dTheta_hub.mean("height")

valid2 = (
    dTheta_surf_mean.notnull() &
    dTheta_hub_mean.notnull()
)

Q1 = ((dTheta_surf_mean > 0) & (dTheta_hub_mean > 0))
Q2 = ((dTheta_surf_mean > 0) & (dTheta_hub_mean < 0))
Q3 = ((dTheta_surf_mean < 0) & (dTheta_hub_mean < 0))
Q4 = ((dTheta_surf_mean < 0) & (dTheta_hub_mean > 0))

# Quadrant Plot:
# plt.figure(figsize=(6,6))
# plt.scatter(dTheta_surf_mean.where(Q1&valid2),dTheta_hub_mean.where(Q1&valid2),color='black',alpha=0.4,label="Coupled Stability")
# plt.scatter(dTheta_surf_mean.where(Q2&valid2),dTheta_hub_mean.where(Q2&valid2),color='blue',alpha=0.4,label="Surface Stable - Hub Unstable")
# plt.scatter(dTheta_surf_mean.where(Q3&valid2),dTheta_hub_mean.where(Q3&valid2),color='gray',alpha=0.4,label="Coupled Instability")
# plt.scatter(dTheta_surf_mean.where(Q4&valid2),dTheta_hub_mean.where(Q4&valid2),color='red',alpha=0.4,label="Surface Unstable - Hub Stable")
# plt.axhline(0,color='k')
# plt.axvline(0,color='k')
# plt.xlabel("dθ/dy (40-60m)")
# plt.ylabel("dθ/dy (120-160m)")
# plt.title("Static Stability Quadrant Analysis")
# plt.legend()
# plt.show()

# Static decoupling occurrences:
static_decoupled = (Q2 | Q4).where(valid2)
staticOverall_percent = 100*static_decoupled.mean()
monthly_percent = 100*(static_decoupled.groupby("time.month").mean())
print(f"{staticOverall_percent.values:.2f}% of the summer (on station) is statically decoupled")

def detect_staticdecoupling(dTheta_surf,dTheta_hub):
    
    valid3 = (
        dTheta_surf.notnull().any(dim="height") &
        dTheta_hub.notnull().any(dim="height")
        )
    
    surf_stable = (dTheta_surf>0).any(dim="height")
    surf_unstable = (dTheta_surf<0).any(dim="height")
    hub_stable = (dTheta_hub>0).any(dim="height")
    hub_unstable = (dTheta_hub<0).any(dim="height")
    
    logic1 = (surf_stable & hub_unstable) & valid3
    logic2 = (surf_unstable & hub_stable) & valid3
    
    times1 = dTheta_surf.time.where(logic1,drop=True)
    times2 = dTheta_surf.time.where(logic2,drop=True)
    
    return times1,times2
        
stimes1,stimes2 = detect_staticdecoupling(dTheta_surf,dTheta_hub)
print(f"statically stable near surface and statically unstable near hub: {stimes1}")
print(f"statically unstable near surface and statically stable near hub: {stimes2}")

# # plot dTheta at surface:
# plt.figure(figsize=(10, 5))
# dTheta_surf.plot(x="time", y="height", cmap="coolwarm")
# ax = plt.gca()
# ax.xaxis.set_major_formatter(mdates.DateFormatter("%H"))  # only show hours
# ax.axvline(sunrise,color="purple",linestyle="--",linewidth=1.5,label='Sunrise')
# ax.axvline(sunset,color="black",linestyle="--",linewidth=1.5,label='Sunset')
# ax.legend(loc="upper right")
# plt.title("dθ/dy (40-60m) on 27 July, 2024")
# plt.xlabel("UTC Time")
# plt.ylabel("Height (m)")
# plt.tight_layout()
# plt.show()

# # plot dTheta at hub:
# plt.figure(figsize=(10, 5))
# dTheta_hub.plot(x="time", y="height", cmap="coolwarm")
# ax = plt.gca()
# ax.xaxis.set_major_formatter(mdates.DateFormatter("%H"))  # only show hours
# ax.axvline(sunrise,color="purple",linestyle="--",linewidth=1.5,label='Sunrise')
# ax.axvline(sunset,color="black",linestyle="--",linewidth=1.5,label='Sunset')
# ax.legend(loc="upper right")
# plt.title("dθ/dy (120-160m) on 27 July, 2024")
# plt.xlabel("UTC Time")
# plt.ylabel("Height (m)")
# plt.tight_layout()
# plt.show()

# grab wind speed, wind direction from combined lidar file
wind_speed = dataLidar["wind_speed"]
wind_direction = np.deg2rad(dataLidar["wind_direction"])

# calculate u and v
uGeo = -wind_speed * np.sin(wind_direction)
vGeo = -wind_speed * np.cos(wind_direction)
sGeo = np.sqrt(uGeo**2+vGeo**2)

# # plot wind:
# fig,ax = plt.subplots(figsize=(10,5))
# T,Z = np.meshgrid(uGeo["time"],uGeo["height"],indexing="ij")
# q = ax.quiver(T[::5],Z[::5],uGeo[::5],vGeo[::5],sGeo[::5])
# plt.colorbar(q,ax=ax,label="Wind Speed (m/s)")
# ax.xaxis.set_major_formatter(mdates.DateFormatter("%H"))  # only show hours
# ax.set_xlabel("UTC Time")
# ax.set_ylabel("Height (m)")
# ax.set_title("Wind Quiver Plot on 27 July, 2024")
# plt.show()

# calculate surface Bulk Richardson number:
# 1) establish height difference
g = 9.81
hsurf_i = 40
hsurf_f = 60
dZ_surf = hsurf_f-hsurf_i
# 2) change in potential temperature
thetasurf_i = theta.sel(height=hsurf_i)
thetasurf_f = theta.sel(height=hsurf_f)
deltaTheta_surf = thetasurf_f - thetasurf_i # K
# 3) change in temperature
tempsurf_i = tempK.sel(height=hsurf_i)
tempsurf_f = tempK.sel(height=hsurf_f)
avgTemp_surf = (tempsurf_i+tempsurf_f)/2 # K
# 4) change in u,v over heights
usurf_i = uGeo.sel(height=hsurf_i)
usurf_f = uGeo.sel(height=hsurf_f)
dU_surf = usurf_f - usurf_i
vsurf_i = vGeo.sel(height=hsurf_i)
vsurf_f = vGeo.sel(height=hsurf_f)
dV_surf = vsurf_f - vsurf_i
# 5) final calculation
num1_surf = g/avgTemp_surf
num2_surf = deltaTheta_surf*dZ_surf
sGeo_surf = (dU_surf**2+dV_surf**2)
num3_surf = num2_surf/sGeo_surf
BulkRi_surf = num1_surf*num3_surf

# calculate hub Bulk Richardson number:
# 1) establish height difference
hhub_i = 120
hhub_f = 160
dZ_hub = hhub_f-hhub_i
# 2) change in potential temperature
thetahub_i = theta.sel(height=hhub_i)
thetahub_f = theta.sel(height=hhub_f)
deltaTheta_hub = thetahub_f - thetahub_i # K
# 3) change in temperature
temphub_i = tempK.sel(height=hhub_i)
temphub_f = tempK.sel(height=hhub_f)
avgTemp_hub = (temphub_i+temphub_f)/2 # K
# 4) change in u,v over heights
uhub_i = uGeo.sel(height=hhub_i)
uhub_f = uGeo.sel(height=hhub_f)
dU_hub = uhub_f - uhub_i
vhub_i = vGeo.sel(height=hhub_i)
vhub_f = vGeo.sel(height=hhub_f)
dV_hub = vhub_f - vhub_i
# 5) final calculation
num1_hub = g/avgTemp_hub
num2_hub = deltaTheta_hub*dZ_hub
sGeo_hub = (dU_hub**2+dV_hub**2)
num3_hub = num2_hub/sGeo_hub
BulkRi_hub = num1_hub*num3_hub

def detect_dynamicdecoupling(BulkRi_surf,BulkRi_hub):
    
    valid4 = (
        BulkRi_surf.notnull() &
        BulkRi_hub.notnull()
        )
    
    
    dsurf_stable = (BulkRi_surf>0.25)
    dsurf_unstable = (BulkRi_surf<0.25)
    dhub_stable = (BulkRi_hub>0.25)
    dhub_unstable = (BulkRi_hub<0.25)
    
    dlogic1 = (dsurf_stable & dhub_unstable) & valid4
    dlogic2 = (dsurf_unstable & dhub_stable) & valid4
    
    dtimes1 = BulkRi_surf.time.where(dlogic1,drop=True)
    dtimes2 = BulkRi_surf.time.where(dlogic2,drop=True)
    
    return dtimes1,dtimes2

dtimes1,dtimes2 = detect_dynamicdecoupling(BulkRi_surf,BulkRi_hub)
print(f"dynamically stable near surface and dynamically unstable near hub: {dtimes1}")
print(f"dynamically unstable near surface and dynamically stable near hub: {dtimes2}")

# wind speed during decoupling histogram
decoupled_ws1 = wind_speed.sel(time=dtimes1).values.flatten() # stable surf, unstable hub
decoupled_ws2 = wind_speed.sel(time=dtimes2).values.flatten() # unstable surf, stable hub
decoupled_ws1 = decoupled_ws1[~np.isnan(decoupled_ws1)]
decoupled_ws2 = decoupled_ws2[~np.isnan(decoupled_ws2)]

plt.hist(decoupled_ws2,bins=80)
plt.xlabel("Wind Speed (m/s)")
plt.ylabel("Number of occurences (n)")
plt.title("Wind speed distribution (unstable surf, stable hub)")
plt.show()

# wind direction during decoupling wind rose/histogram
decoupled_wd1 = np.rad2deg(wind_direction.sel(time=dtimes1).values.flatten())
decoupled_wd2 = np.rad2deg(wind_direction.sel(time=dtimes2).values.flatten())
decoupled_wd1 = decoupled_wd1[~np.isnan(decoupled_wd1)]
decoupled_wd2 = decoupled_wd2[~np.isnan(decoupled_wd2)]

dir_bins = np.arange(0,361,30)
counts, _ = np.histogram(decoupled_wd2,bins=dir_bins)
freq = counts / counts.sum() * 100
rose_theta = np.deg2rad(dir_bins[:-1])
rose_width = np.deg2rad(30)
fig = plt.figure(figsize=(6,6))
ax = plt.subplot(111,polar=True)
ax.bar(rose_theta,freq,width=rose_width,bottom=0,label="Frequency (%)")
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)
ax.set_title("Wind direction distribution (unstable surf, stable hub)")
plt.show()

# time of day during decoupling histogram
decoupled_times1 = dtimes1.dt.hour + dtimes1.dt.minute/60
decoupled_times2 = dtimes2.dt.hour + dtimes2.dt.minute/60

plt.hist(decoupled_times2,bins=24)
plt.xlabel("UTC Time")
plt.xlim(0,24)
plt.xticks([2,4,6,8,10,12,14,16,18,20,22,24])
plt.ylabel("Number of occurences (n)")
plt.title("Occurrences throughout the day (unstable surf, stable hub)")
plt.show()

# wind rose
speed_bins = [0,2,4,6,8,10,12,14,16]
H,dir_edges,speed_edges = np.histogram2d(decoupled_wd2,decoupled_ws2,bins=[dir_bins,speed_bins])
freq = H / H.sum()
bottom = np.zeros(len(rose_theta))
fig = plt.figure(figsize=(6,6))
ax = plt.subplot(111,polar=True)
colors = plt.cm.viridis(np.linspace(0,1,len(speed_bins)-1))
for i in range(len(speed_bins)-1):
    values = freq[:, i]
    bars = ax.bar(rose_theta, values, width=rose_width, bottom=bottom, color=colors[i], label=f"{speed_bins[i]}-{speed_bins[i+1]} m/s")
    bottom += values
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)
ax.set_title("Wind speed and direction (Surface Unstable - Hub Stable)")
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.show()

# wind rose for each time of day
decoupled_ws2 = wind_speed.sel(time=dtimes2)
decoupled_wd2 = np.rad2deg(wind_direction.sel(time=dtimes2))

night = (decoupled_times2 > 1) & (decoupled_times2 < 7)
sunrising = (decoupled_times2 >= 7) & (decoupled_times2 <= 13)
day = (decoupled_times2 > 13) & (decoupled_times2 < 19)
sunsetting = (decoupled_times2 >= 19) | (decoupled_times2 <= 1)

peak = (decoupled_times2 > 13) & (decoupled_times2 < 23)

night_ws = decoupled_ws2.where(night)
night_wd = decoupled_wd2.where(night)
sunrise_ws = decoupled_ws2.where(sunrising)
sunrise_wd = decoupled_wd2.where(sunrising)
day_ws = decoupled_ws2.where(day)
day_wd = decoupled_wd2.where(day)
sunset_ws = decoupled_ws2.where(sunsetting)
sunset_wd = decoupled_wd2.where(sunsetting)

peak_ws = decoupled_ws2.where(peak)
peak_wd = decoupled_wd2.where(peak)

night_ws = night_ws.values.flatten()
night_wd = night_wd.values.flatten()
sunrise_ws = sunrise_ws.values.flatten()
sunrise_wd = sunrise_wd.values.flatten()
day_ws = day_ws.values.flatten()
day_wd = day_wd.values.flatten()
sunset_ws = sunset_ws.values.flatten()
sunset_wd = sunset_wd.values.flatten()

peak_ws = peak_ws.values.flatten()
peak_wd = peak_wd.values.flatten()

H,dir_edges,speed_edges = np.histogram2d(sunset_wd,sunset_ws,bins=[dir_bins,speed_bins])
freq = H / H.sum()
fig = plt.figure(figsize=(6,6))
ax = plt.subplot(111,polar=True)
colors = plt.cm.viridis(np.linspace(0,1,len(speed_bins)-1))
for i in range(len(speed_bins)-1):
    values = freq[:, i]
    bars = ax.bar(rose_theta, values, width=rose_width, bottom=bottom, color=colors[i], label=f"{speed_bins[i]}-{speed_bins[i+1]} m/s")
    bottom += values
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)
ax.set_title("Wind speed and direction from 1900-0100 UTC (unstable surf, stable hub)")
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.show()

# coupled cases mask
all_times = wind_speed.time
decoupTimes = dtimes1.to_index().union(dtimes2.to_index())
decoupMask = all_times.isin(decoupTimes)
coupledMask = ~ decoupMask
coupled_times = all_times.where(coupledMask,drop=True)

# plot coupled cases
coupled_ws = wind_speed.sel(time=coupled_times).values.flatten()
coupled_ws = coupled_ws[~np.isnan(coupled_ws)]
coupled_wd = np.rad2deg(wind_direction.sel(time=coupled_times).values.flatten())
coupled_wd = coupled_wd[~np.isnan(coupled_wd)]

H,dir_edges,speed_edges = np.histogram2d(coupled_wd,coupled_ws,bins=[dir_bins,speed_bins])
freq = H / H.sum()
fig = plt.figure(figsize=(6,6))
ax = plt.subplot(111,polar=True)
colors = plt.cm.viridis(np.linspace(0,1,len(speed_bins)-1))
for i in range(len(speed_bins)-1):
    values = freq[:, i]
    bars = ax.bar(rose_theta, values, width=rose_width, bottom=bottom, color=colors[i], label=f"{speed_bins[i]}-{speed_bins[i+1]} m/s")
    bottom += values
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)
ax.set_title("Wind speed and direction (Coupled)")
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.show()

plt.hist(coupled_ws,bins=50)
plt.xlabel("Wind speed (m/s)")
plt.xlim(0,17.5)
plt.xticks([0,2.5,5.0,7.5,10.0,12.5,15.0,17.5])
plt.ylabel("Number of occurences (n)")
plt.title("Wind speed distribution (coupled cases)")
plt.show()

coupledTimes = coupled_times.dt.hour + coupled_times.dt.minute/60
night = (coupledTimes > 1) & (coupledTimes < 7)
sunrising = (coupledTimes >= 7) & (coupledTimes <= 13)
day = (coupledTimes > 13) & (coupledTimes < 19)
sunsetting = (coupledTimes >= 19) | (coupledTimes <= 1)

coupled_ws = wind_speed.sel(time=coupled_times)
coupled_wd = np.rad2deg(wind_direction.sel(time=coupled_times))

nightWS = coupled_ws.where(night,drop=True).values.flatten()
nightWD = coupled_wd.where(night,drop=True).values.flatten()

H,dir_edges,speed_edges = np.histogram2d(nightWD,nightWS,bins=[dir_bins,speed_bins])
freq = H / H.sum()
fig = plt.figure(figsize=(6,6))
ax = plt.subplot(111,polar=True)
colors = plt.cm.viridis(np.linspace(0,1,len(speed_bins)-1))
for i in range(len(speed_bins)-1):
    values = freq[:, i]
    bars = ax.bar(rose_theta, values, width=rose_width, bottom=bottom, color=colors[i], label=f"{speed_bins[i]}-{speed_bins[i+1]} m/s")
    bottom += values
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)
ax.set_title("Wind speed and direction from 0100-0700 UTC (Coupled cases)")
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.show()

plt.hist(nightWS,bins=50)
plt.xlabel("Wind speed (m/s)")
plt.xlim(0,17.5)
plt.xticks([0,2.5,5.0,7.5,10.0,12.5,15.0,17.5])
plt.ylabel("Number of occurences (n)")
plt.title("Wind speed distribution 0100-0700 UTC (coupled cases)")
plt.show()

# # plot surface Bulk Richardson number:
# fig, ax = plt.subplots(figsize=(6,5))
# BulkRi_surf.plot(ax=ax)
# ax.set_xlim(BulkRi_surf.time.min().values,BulkRi_surf.time.max().values)
# ax.xaxis.set_major_formatter(mdates.DateFormatter("%H"))
# ax.axhline(0.25, linestyle="--", label='Critical Ri')
# ax.axvline(sunrise, linestyle="--", label='Sunrise')
# ax.axvline(sunset, linestyle="--", label='Sunset')
# ax.set_title("27 July 2024 (40-60m)")
# ax.set_xlabel("UTC Time")
# ax.set_ylabel("Bulk Richardson number")
# ax.legend()
# plt.tight_layout()
# plt.show()

# # plot hub height Bulk Richardson number:
# fig, ax = plt.subplots(figsize=(6,5))
# BulkRi_hub.plot(ax=ax)
# ax.set_xlim(BulkRi_hub.time.min().values,BulkRi_hub.time.max())
# ax.xaxis.set_major_formatter(mdates.DateFormatter("%H"))
# ax.axhline(0.25, linestyle="--", label='Critical Ri')
# ax.axvline(sunrise, linestyle="--", label='Sunrise')
# ax.axvline(sunset, linestyle="--", label='Sunset')
# ax.set_title("27 July 2024 (120-160m)")
# ax.set_xlabel("UTC Time")
# ax.set_ylabel("Bulk Richardson number")
# ax.legend()
# plt.tight_layout()
# plt.show()

# dynamic stability quadrant analysis: [IN PROGRESS]
inRange = (np.abs(BulkRi_surf<1)) & (np.abs(BulkRi_hub<1)) # this can be adjusted

valid4 = (
    BulkRi_surf.notnull() &
    BulkRi_hub.notnull()
    )

Q1 = (BulkRi_surf > 0.25) & (BulkRi_hub > 0.25)
Q2 = (BulkRi_surf > 0.25) & (BulkRi_hub < 0.25)
Q3 = (BulkRi_surf < 0.25) & (BulkRi_hub < 0.25)
Q4 = (BulkRi_surf < 0.25) & (BulkRi_hub > 0.25)

plt.figure(figsize=(6,6))
plt.scatter(BulkRi_surf.where(Q1&valid4&inRange),BulkRi_hub.where(Q1&valid4&inRange),color='black',alpha=0.4,label="Coupled Stability")
plt.scatter(BulkRi_surf.where(Q2&valid4&inRange),BulkRi_hub.where(Q2&valid4&inRange),color='blue',alpha=0.4,label="Surface Stable - Hub Turbulent")
plt.scatter(BulkRi_surf.where(Q3&valid4&inRange),BulkRi_hub.where(Q3&valid4&inRange),color='gray',alpha=0.4,label="Coupled Turbulence")
plt.scatter(BulkRi_surf.where(Q4&valid4&inRange),BulkRi_hub.where(Q4&valid4&inRange),color='red',alpha=0.4,label="Surface Turbulent - Hub Stable")
plt.axhline(0.25,color='k')
plt.axvline(0.25,color='k')
plt.xlabel("Ri_B (40-60m)")
plt.ylabel("Ri_B (120-160m)")
plt.title("Dynamic Stability Quadrant Analysis")
plt.legend()
plt.show()

# decoupling percentages:
decoupled = (Q2 | Q4).where(valid4)
overall_percent = 100*decoupled.mean()
monthly_percent = 100*(decoupled.groupby("time.month").mean())

# # frequency along time:
# months = xr.DataArray(["May", "Jun", "Jul", "Aug", "Sep"])
# plt.figure(figsize=(8,5))
# plt.bar(months,monthly_percent.values,width=0.8)
# plt.ylim((0,100))
# plt.xlabel("UTC Time")
# plt.ylabel("Frequency (%)")
# plt.title("Dynamic Stability Decoupling (Summer 2024)")
# plt.show()

print(f"{overall_percent.values:.2f}% of the summer (on station) is dynamically decoupled")

# # identify long durations (1+ hours):
# decouple_flag = decoupled.astype(int)
# groups = (decouple_flag.diff("time") != 0).cumsum("time")
# for num in np.unique(groups):
#     segment = decouple_flag.where(groups==num,drop=True)
#     if segment.mean() == 1:
#         duration = len(segment)
#         if duration >= 2:
#             print(segment.time.values[0], "to", segment.time.values[-1])




