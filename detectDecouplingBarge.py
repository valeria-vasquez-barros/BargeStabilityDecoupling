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
import seaborn as sb

#%% Data initialization
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

#%% ASSIST variables, static stability analysis

# Grab theta, temp variables from combined assist file
theta = dataAssist["theta"]
# .sel(time=slice("2024-06-29 00:00:00","2024-06-29 23:59:59"))
temp = dataAssist["temperature"]
# .sel(time=slice("2024-06-29 00:00:00","2024-06-29 23:59:59"))
tempK = temp + 273.15 # convert to K

# dTheta gradient
dTheta = theta.differentiate("height")
dTheta_surf = dTheta.sel(height=slice(40,60)) 
dTheta_hub = dTheta.sel(height=slice(120,160))
dTheta_times = dTheta.time

# Calculate bulk deltaTheta/deltaz "near-surface" and "hub-height"
thetasurf_i = theta.sel(height=40)
thetasurf_f = theta.sel(height=60)
deltaTheta_surf = (thetasurf_f - thetasurf_i)/20 # K

thetahub_i = theta.sel(height=120)
thetahub_f = theta.sel(height=160)
deltaTheta_hub = (thetahub_f - thetahub_i)/40 # K

# Static decoupling occurrences:
def detect_staticdecoupling(deltaTheta_surf,deltaTheta_hub):
    
    valid2 = (
        deltaTheta_surf.notnull() &
        deltaTheta_hub.notnull()
        )
    
    surf_stable = (deltaTheta_surf>0)
    surf_unstable = (deltaTheta_surf<0)
    hub_stable = (deltaTheta_hub>0)
    hub_unstable = (deltaTheta_hub<0)
    
    logic1 = (surf_stable & hub_unstable) & valid2
    logic2 = (surf_unstable & hub_stable) & valid2
    
    stimes1 = deltaTheta_surf.time.where(logic1,drop=True)
    stimes2 = deltaTheta_hub.time.where(logic2,drop=True)
    
    return stimes1,stimes2
        
stimes1,stimes2 = detect_staticdecoupling(deltaTheta_surf,deltaTheta_hub)
# print(f"statically stable near surface and statically unstable near hub: {stimes1}")
# print(f"statically unstable near surface and statically stable near hub: {stimes2}")

# Static Quadrant Plot:
valid2 = (
    deltaTheta_surf.notnull() & # remove off-station and no data
    deltaTheta_hub.notnull()
)

Q1 = ((deltaTheta_surf > 0) & (deltaTheta_hub > 0))
Q2 = ((deltaTheta_surf > 0) & (deltaTheta_hub < 0))
Q3 = ((deltaTheta_surf < 0) & (deltaTheta_hub < 0))
Q4 = ((deltaTheta_surf < 0) & (deltaTheta_hub > 0))

Q1percent = Q1.where(valid2).mean()*100
# print(f"Q1:{Q1percent.values:.2f}%")
Q2percent = Q2.where(valid2).mean()*100
# print(f"Q2:{Q2percent.values:.2f}%")
Q3percent = Q3.where(valid2).mean()*100
# print(f"Q3:{Q3percent.values:.2f}%")
Q4percent = Q4.where(valid2).mean()*100
# print(f"Q4:{Q4percent.values:.2f}%")

# plt.figure(figsize=(6,6))
# plt.scatter(deltaTheta_surf.where(Q1&valid2),deltaTheta_hub.where(Q1&valid2),color='blue',alpha=0.4,label="Coupled Stability")
# plt.scatter(deltaTheta_surf.where(Q2&valid2),deltaTheta_hub.where(Q2&valid2),color='gray',alpha=0.4,label="Surface Stable - Hub Unstable")
# plt.scatter(deltaTheta_surf.where(Q3&valid2),deltaTheta_hub.where(Q3&valid2),color='red',alpha=0.4,label="Coupled Instability")
# plt.scatter(deltaTheta_surf.where(Q4&valid2),deltaTheta_hub.where(Q4&valid2),color='purple',alpha=0.4,label="Surface Unstable - Hub Stable")
# plt.axhline(0,color='k')
# plt.axvline(0,color='k')
# plt.text(0.8,0.9,f"{Q1percent.values:.1f}%",transform=plt.gca().transAxes,fontweight="bold")
# plt.text(0.8,0.3,f"{Q2percent.values:.1f}%",transform=plt.gca().transAxes,fontweight="bold")
# plt.text(0.1,0.1,f"{Q3percent.values:.1f}%",transform=plt.gca().transAxes,fontweight="bold")
# plt.text(0.1,0.9,f"{Q4percent.values:.1f}%",transform=plt.gca().transAxes,fontweight="bold")
# plt.xlabel("dθ/dz (40-60m)")
# plt.ylabel("dθ/dz (120-160m)")
# # plt.title("Static Stability Quadrant Analysis")
# plt.legend()
# plt.show()

static_decoupled = (Q2 | Q4).where(valid2) # exclude null values (off-station or no data)
staticOverall_percent = 100*static_decoupled.mean() # mean considers total (non-null)
monthly_num = (static_decoupled.groupby("time.month").sum())
monthly_percent = (static_decoupled.groupby("time.month").mean())
print(f"{staticOverall_percent.values:.2f}% of the summer (on station) is statically decoupled")

# # frequency along time:
# months = xr.DataArray(["Jun", "Jul", "Aug", "Sep"])
# plt.figure(figsize=(8,5))
# plt.bar(months,monthly_num.sel(month=slice(6,9)).values,width=0.8)
# plt.xlabel("UTC Time")
# plt.ylabel("Number of Occurrences (n)")
# # plt.title("Static Stability Decoupling (Summer 2024)")
# plt.show()

# # identify long durations (1+ hours):
# decouple_flag = static_decoupled.astype(int)
# groups = (decouple_flag.diff("time") != 0).cumsum("time")
# for num in np.unique(groups):
#     segment = decouple_flag.where(groups==num,drop=True)
#     if segment.mean() == 1:
#         duration = len(segment)
#         if duration >= 6:
#             print(segment.time.values[0], "to", segment.time.values[-1])

# # try plot of dTheta across all heights for clarity
# plt.figure(figsize=(10, 5))
# dTheta.sel(height=slice(40,200)).plot(x="time", y="height", cmap="coolwarm")
# ax = plt.gca()
# ax.xaxis.set_major_formatter(mdates.DateFormatter("%H"))  # only show hours
# ax.axvline(sunrise,color="purple",linestyle="--",linewidth=1.5,label='Sunrise')
# ax.axvline(sunset,color="black",linestyle="--",linewidth=1.5,label='Sunset')
# # ax.legend(loc="upper right",label="dθ/dz")
# # plt.title("dθ/dz")
# plt.xlabel("UTC Time")
# plt.ylabel("Height (m)")
# plt.tight_layout()
# plt.show()

# # plot dTheta at surface:
# plt.figure(figsize=(10, 5))
# dTheta_surf.plot(x="time", y="height", cmap="coolwarm")
# ax = plt.gca()
# ax.xaxis.set_major_formatter(mdates.DateFormatter("%H"))  # only show hours
# ax.axvline(sunrise,color="purple",linestyle="--",linewidth=1.5,label='Sunrise')
# ax.axvline(sunset,color="black",linestyle="--",linewidth=1.5,label='Sunset')
# ax.legend(loc="upper right")
# plt.title("dθ/dz (40-60m)")
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
# plt.title("dθ/dz (120-160m)")
# plt.xlabel("UTC Time")
# plt.ylabel("Height (m)")
# plt.tight_layout()
# plt.show()

#%% Lidar variables

# grab wind speed, wind direction from combined lidar file
wind_speed = dataLidar["wind_speed"]
# .sel(time=slice("2024-06-29 00:00:00","2024-06-29 23:59:59"))
wind_direction = np.deg2rad(dataLidar["wind_direction"])
# .sel(time=slice("2024-06-29 00:00:00","2024-06-29 23:59:59"))

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
# ax.set_title("Wind Quiver Plot")
# plt.show()

#%% Bulk Richardson number, dynamic stability analysis

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
BulkRi_surf = BulkRi_surf.where(sGeo_surf>0)

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
BulkRi_hub = BulkRi_hub.where(sGeo_hub>0)

def detect_dynamicdecoupling(BulkRi_surf,BulkRi_hub):
    
    valid4 = (
        BulkRi_surf.notnull() &
        BulkRi_hub.notnull()
        )
    
    dsurf_stable = (BulkRi_surf>0)
    dsurf_unstable = (BulkRi_surf<0)
    dhub_stable = (BulkRi_hub>0)
    dhub_unstable = (BulkRi_hub<0)
    
    dlogic1 = (dsurf_stable & dhub_unstable) & valid4
    dlogic2 = (dsurf_unstable & dhub_stable) & valid4
    
    dtimes1 = BulkRi_surf.time.where(dlogic1,drop=True)
    dtimes2 = BulkRi_surf.time.where(dlogic2,drop=True)
    
    return dtimes1,dtimes2

dtimes1,dtimes2 = detect_dynamicdecoupling(BulkRi_surf,BulkRi_hub)
# print(f"dynamically stable near surface and dynamically unstable near hub: {dtimes1}")
# print(f"dynamically unstable near surface and dynamically stable near hub: {dtimes2}")

# # Dynamic Quadrant Plot:
# valid4 = (
#     BulkRi_surf.notnull() &
#     BulkRi_hub.notnull()
#     )

# Q1 = (BulkRi_surf > 0) & (BulkRi_hub > 0)
# Q2 = (BulkRi_surf > 0) & (BulkRi_hub < 0)
# Q3 = (BulkRi_surf < 0) & (BulkRi_hub < 0)
# Q4 = (BulkRi_surf < 0) & (BulkRi_hub > 0)

# Q1percent = Q1.where(valid4).mean()*100
# print(f"Q1:{Q1percent.values:.2f}%")
# Q2percent = Q2.where(valid4).mean()*100
# print(f"Q2:{Q2percent.values:.2f}%")
# Q3percent = Q3.where(valid4).mean()*100
# print(f"Q3:{Q3percent.values:.2f}%")
# Q4percent = Q4.where(valid4).mean()*100
# print(f"Q4:{Q4percent.values:.2f}%")

# plt.figure(figsize=(6,6))
# plt.scatter(BulkRi_surf.where(Q1&valid4),BulkRi_hub.where(Q1&valid4),color='blue',alpha=0.4,label="Coupled Stability")
# plt.scatter(BulkRi_surf.where(Q2&valid4),BulkRi_hub.where(Q2&valid4),color='gray',alpha=0.4,label="Surface Stable - Hub Turbulent")
# plt.scatter(BulkRi_surf.where(Q3&valid4),BulkRi_hub.where(Q3&valid4),color='red',alpha=0.4,label="Coupled Turbulence")
# plt.scatter(BulkRi_surf.where(Q4&valid4),BulkRi_hub.where(Q4&valid4),color='purple',alpha=0.4,label="Surface Turbulent - Hub Stable")
# plt.axhline(0,color='k')
# plt.axvline(0,color='k')
# plt.xlim([-100,100])
# plt.ylim([-100,100])
# plt.text(0.8,0.9,f"{Q1percent.values:.1f}%",transform=plt.gca().transAxes,fontweight="bold")
# plt.text(0.8,0.1,f"{Q2percent.values:.1f}%",transform=plt.gca().transAxes,fontweight="bold")
# plt.text(0.1,0.3,f"{Q3percent.values:.1f}%",transform=plt.gca().transAxes,fontweight="bold")
# plt.text(0.1,0.9,f"{Q4percent.values:.1f}%",transform=plt.gca().transAxes,fontweight="bold")
# plt.xlabel("Ri_B (40-60m)")
# plt.ylabel("Ri_B (120-160m)")
# plt.title("Dynamic Stability Quadrant Analysis")
# plt.legend()
# plt.show()

# decoupled = (Q2 | Q4).where(valid4)
# overall_percent = 100*decoupled.mean()
# monthly_num = (decoupled.groupby("time.month").sum())
# monthly_percent = 100*(decoupled.groupby("time.month").mean())
# print(f"{overall_percent.values:.2f}% of the summer (on station) is dynamically decoupled")

# decoupled_percents = xr.DataArray([17.80795344,25.41222114,40.85354025,57.90494665,41.02812803])
# cutoffs = xr.DataArray(["-1", "0", "0.1", "0.25", "1"])
# plt.figure(figsize=(8,5))
# plt.bar(cutoffs,decoupled_percents.values,width=0.8)
# plt.ylim((0,100))
# plt.xlabel("Critical Richardson Number")
# plt.ylabel("Dynamic Decoupling Frequency (%)")
# plt.title("Critical Richardson Number Sensitivity Analysis")
# plt.show()

# # frequency along time:
# months = xr.DataArray(["Jun", "Jul", "Aug", "Sep"])
# plt.figure(figsize=(8,5))
# plt.bar(months,monthly_num.sel(month=slice(6,9)).values,width=0.8)
# plt.xlabel("UTC Time")
# plt.ylabel("Number of Occurrences (n)")
# # plt.title("Dynamic Stability Decoupling (Summer 2024)")
# plt.show()

# # identify long durations (1+ hours):
# decouple_flag = decoupled.astype(int)
# groups = (decouple_flag.diff("time") != 0).cumsum("time")
# for num in np.unique(groups):
#     segment = decouple_flag.where(groups==num,drop=True)
#     if segment.mean() == 1:
#         duration = len(segment)
#         if duration >= 6:
#             print(segment.time.values[0], "to", segment.time.values[-1])

# # plot surface Bulk Richardson number:
# fig, ax = plt.subplots(figsize=(6,5))
# BulkRi_surf.plot(ax=ax)
# ax.set_xlim(BulkRi_surf.time.min().values,BulkRi_surf.time.max().values)
# ax.xaxis.set_major_formatter(mdates.DateFormatter("%H"))
# ax.axhline(0, linestyle="--", label='Critical Ri')
# # ax.axvline(sunrise, linestyle="--", label='Sunrise')
# # ax.axvline(sunset, linestyle="--", label='Sunset')
# # ax.set_title("40-60m")
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
# ax.axhline(0, linestyle="--", label='Critical Ri')
# # ax.axvline(sunrise, linestyle="--", label='Sunrise')
# # ax.axvline(sunset, linestyle="--", label='Sunset')
# # ax.set_title("120-160m")
# ax.set_xlabel("UTC Time")
# ax.set_ylabel("Bulk Richardson number")
# ax.legend()
# plt.tight_layout()
# plt.show()

#%% Summary statistics of decoupled events

# # grab wind speed/direction and times when decoupling occurs
# decoupled_ws1 = wind_speed.sel(time=dtimes1).values.flatten() # stable surf, unstable hub
# decoupled_ws2 = wind_speed.sel(time=dtimes2).values.flatten() # unstable surf, stable hub
# decoupled_ws1 = decoupled_ws1[~np.isnan(decoupled_ws1)]
# decoupled_ws2 = decoupled_ws2[~np.isnan(decoupled_ws2)]

# decoupled_wd1 = np.rad2deg(wind_direction.sel(time=dtimes1).values.flatten())
# decoupled_wd2 = np.rad2deg(wind_direction.sel(time=dtimes2).values.flatten())
# decoupled_wd1 = decoupled_wd1[~np.isnan(decoupled_wd1)]
# decoupled_wd2 = decoupled_wd2[~np.isnan(decoupled_wd2)]

# decoupled_times1 = dtimes1.dt.hour + dtimes1.dt.minute/60
# decoupled_times2 = dtimes2.dt.hour + dtimes2.dt.minute/60

# # Wind speed (Q4) distribution histogram
# plt.hist(decoupled_ws2,bins=80)
# plt.xlabel("Wind Speed (m/s)")
# plt.ylabel("Number of occurences (n)")
# plt.title("Wind speed distribution (unstable surf, stable hub)")
# plt.show()

# # Wind rose (Q4)
# dir_bins = np.arange(0,361,30)
# counts, _ = np.histogram(decoupled_wd2,bins=dir_bins)
# freq = counts / counts.sum() * 100
# rose_theta = np.deg2rad(dir_bins[:-1])
# rose_width = np.deg2rad(30)
# speed_bins = [0,2,4,6,8,10,12,14,16,18,20,22,24]

# H,dir_edges,speed_edges = np.histogram2d(decoupled_wd2,decoupled_ws2,bins=[dir_bins,speed_bins])
# freq = H / H.sum()
# bottom = np.zeros(len(rose_theta))
# fig = plt.figure(figsize=(6,6))
# ax = plt.subplot(111,polar=True)
# colors = plt.cm.viridis(np.linspace(0,1,len(speed_bins)-1))
# for i in range(len(speed_bins)-1):
#     values = freq[:, i]
#     bars = ax.bar(rose_theta, values, width=rose_width, bottom=bottom, color=colors[i], label=f"{speed_bins[i]}-{speed_bins[i+1]} m/s")
#     bottom += values
# ax.set_theta_zero_location("N")
# ax.set_theta_direction(-1)
# # ax.set_title("Wind Rose (Surface Turbulent - Hub Stable)")
# ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
# plt.show()

# # Decoupling (Q4) histogram by time of day
# plt.hist(decoupled_times2,bins=24)
# plt.xlabel("UTC Time")
# plt.xlim(0,24)
# plt.xticks([2,4,6,8,10,12,14,16,18,20,22,24])
# plt.ylabel("Number of occurences (n)")
# plt.title("Occurrences throughout the day (Surface Turbulent - Hub Stable)")
# plt.show()

# # Wind rose (Q4) for each time of day
# decoupled_ws2 = wind_speed.sel(time=dtimes2)
# decoupled_wd2 = np.rad2deg(wind_direction.sel(time=dtimes2))

# night = (decoupled_times2 > 1) & (decoupled_times2 < 7)
# sunrising = (decoupled_times2 >= 7) & (decoupled_times2 <= 13)
# day = (decoupled_times2 > 13) & (decoupled_times2 < 19)
# sunsetting = (decoupled_times2 >= 19) | (decoupled_times2 <= 1)

# peak = (decoupled_times2 > 13) & (decoupled_times2 < 23)

# night_ws = decoupled_ws2.where(night)
# night_ws = night_ws.values.flatten()
# night_ws = night_ws[~np.isnan(night_ws)]
# night_wd = decoupled_wd2.where(night)
# night_wd = night_wd.values.flatten()
# night_wd = night_wd[~np.isnan(night_wd)]

# sunrise_ws = decoupled_ws2.where(sunrising)
# sunrise_ws = sunrise_ws.values.flatten()
# sunrise_ws = sunrise_ws[~np.isnan(sunrise_ws)]
# sunrise_wd = decoupled_wd2.where(sunrising)
# sunrise_wd = sunrise_wd.values.flatten()
# sunrise_wd = sunrise_wd[~np.isnan(sunrise_wd)]

# day_ws = decoupled_ws2.where(day)
# day_ws = day_ws.values.flatten()
# day_ws = day_ws[~np.isnan(day_ws)]
# day_wd = decoupled_wd2.where(day)
# day_wd = day_wd.values.flatten()
# day_wd = day_wd[~np.isnan(day_wd)]

# sunset_ws = decoupled_ws2.where(sunsetting)
# sunset_ws = sunset_ws.values.flatten()
# sunset_ws = sunset_ws[~np.isnan(sunset_ws)]
# sunset_wd = decoupled_wd2.where(sunsetting)
# sunset_wd = sunset_wd.values.flatten()
# sunset_wd = sunset_wd[~np.isnan(sunset_wd)]

# peak_ws = decoupled_ws2.where(peak)
# peak_ws = peak_ws.values.flatten()
# peak_ws = peak_ws[~np.isnan(peak_ws)]
# peak_wd = decoupled_wd2.where(peak)
# peak_wd = peak_wd.values.flatten()
# peak_wd = peak_wd[~np.isnan(peak_wd)]

# H,dir_edges,speed_edges = np.histogram2d(night_wd,night_ws,bins=[dir_bins,speed_bins])
# freq = H/H.sum()
# bottom = np.zeros(len(rose_theta))
# fig = plt.figure(figsize=(6,6))
# ax = plt.subplot(111,polar=True)
# colors = plt.cm.viridis(np.linspace(0,1,len(speed_bins)-1))
# for i in range(len(speed_bins)-1):
#     values = freq[:, i]
#     bars = ax.bar(rose_theta, values, width=rose_width, bottom=bottom, color=colors[i], label=f"{speed_bins[i]}-{speed_bins[i+1]} m/s")
#     bottom += values
# ax.set_theta_zero_location("N")
# ax.set_theta_direction(-1)
# ax.set_title("Wind rose from 0100-0700 UTC (Surface Turbulent - Hub Stable)")
# ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
# plt.show()

# H,dir_edges,speed_edges = np.histogram2d(sunrise_wd,sunrise_ws,bins=[dir_bins,speed_bins])
# freq = H/H.sum()
# bottom = np.zeros(len(rose_theta))
# fig = plt.figure(figsize=(6,6))
# ax = plt.subplot(111,polar=True)
# colors = plt.cm.viridis(np.linspace(0,1,len(speed_bins)-1))
# for i in range(len(speed_bins)-1):
#     values = freq[:, i]
#     bars = ax.bar(rose_theta, values, width=rose_width, bottom=bottom, color=colors[i], label=f"{speed_bins[i]}-{speed_bins[i+1]} m/s")
#     bottom += values
# ax.set_theta_zero_location("N")
# ax.set_theta_direction(-1)
# ax.set_title("Wind rose from 0700-1300 UTC (Surface Turbulent, Hub Stable)")
# ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
# plt.show()

# H,dir_edges,speed_edges = np.histogram2d(day_wd,day_ws,bins=[dir_bins,speed_bins])
# freq = H/H.sum()
# bottom = np.zeros(len(rose_theta))
# fig = plt.figure(figsize=(6,6))
# ax = plt.subplot(111,polar=True)
# colors = plt.cm.viridis(np.linspace(0,1,len(speed_bins)-1))
# for i in range(len(speed_bins)-1):
#     values = freq[:, i]
#     bars = ax.bar(rose_theta, values, width=rose_width, bottom=bottom, color=colors[i], label=f"{speed_bins[i]}-{speed_bins[i+1]} m/s")
#     bottom += values
# ax.set_theta_zero_location("N")
# ax.set_theta_direction(-1)
# ax.set_title("Wind rose from 1300-1900 UTC (Surface Turbulent, Hub Stable)")
# ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
# plt.show()

# H,dir_edges,speed_edges = np.histogram2d(sunset_wd,sunset_ws,bins=[dir_bins,speed_bins])
# freq = H/H.sum()
# bottom = np.zeros(len(rose_theta))
# fig = plt.figure(figsize=(6,6))
# ax = plt.subplot(111,polar=True)
# colors = plt.cm.viridis(np.linspace(0,1,len(speed_bins)-1))
# for i in range(len(speed_bins)-1):
#     values = freq[:, i]
#     bars = ax.bar(rose_theta, values, width=rose_width, bottom=bottom, color=colors[i], label=f"{speed_bins[i]}-{speed_bins[i+1]} m/s")
#     bottom += values
# ax.set_theta_zero_location("N")
# ax.set_theta_direction(-1)
# ax.set_title("Wind rose from 1900-0100 UTC (Surface Turbulent, Hub Stable)")
# ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
# plt.show()

# # Look at decoupled wind roses by surface and hub height
# surf_ws2 = wind_speed.sel(height=slice(40,60),time=dtimes2).values.flatten()
# surf_ws2 = surf_ws2[~np.isnan(surf_ws2)]
# surf_wd2 = np.rad2deg(wind_direction.sel(height=slice(40,60),time=dtimes2).values.flatten())
# surf_wd2 = surf_wd2[~np.isnan(surf_wd2)]

# hub_ws2 = wind_speed.sel(height=slice(120,160),time=dtimes2).values.flatten()
# hub_ws2 = hub_ws2[~np.isnan(hub_ws2)]
# hub_wd2 = np.rad2deg(wind_direction.sel(height=slice(120,160),time=dtimes2).values.flatten())
# hub_wd2 = hub_wd2[~np.isnan(hub_wd2)]

# dir_bins = np.arange(0,361,30)
# counts, _ = np.histogram(decoupled_wd2,bins=dir_bins)
# freq = counts / counts.sum() * 100
# rose_theta = np.deg2rad(dir_bins[:-1])
# rose_width = np.deg2rad(30)
# speed_bins = [0,2,4,6,8,10,12,14,16]

# H,dir_edges,speed_edges = np.histogram2d(surf_wd2,surf_ws2,bins=[dir_bins,speed_bins])
# freq = H / H.sum()
# bottom = np.zeros(len(rose_theta))
# fig = plt.figure(figsize=(6,6))
# ax = plt.subplot(111,polar=True)
# colors = plt.cm.viridis(np.linspace(0,1,len(speed_bins)-1))
# for i in range(len(speed_bins)-1):
#     values = freq[:, i]
#     bars = ax.bar(rose_theta, values, width=rose_width, bottom=bottom, color=colors[i], label=f"{speed_bins[i]}-{speed_bins[i+1]} m/s")
#     bottom += values
# ax.set_theta_zero_location("N")
# ax.set_theta_direction(-1)
# # ax.set_title("Surface-level Wind Rose (Surface Turbulent - Hub Stable)")
# ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
# plt.show()

#%% Summary statistics for coupled events

# # Coupled cases mask
# all_times = wind_speed.time
# decoupTimes = dtimes1.to_index().union(dtimes2.to_index())
# decoupMask = all_times.isin(decoupTimes)
# coupledMask = ~ decoupMask
# coupled_times = all_times.where(coupledMask,drop=True)

# # grab wind speed/direction when coupled cases occur
# coupled_ws = wind_speed.sel(time=coupled_times).values.flatten()
# coupled_ws = coupled_ws[~np.isnan(coupled_ws)]
# coupled_wd = np.rad2deg(wind_direction.sel(time=coupled_times).values.flatten())
# coupled_wd = coupled_wd[~np.isnan(coupled_wd)]

# # Wind speed (coupled) distribution histogram
# plt.hist(coupled_ws,bins=50)
# plt.xlabel("Wind speed (m/s)")
# plt.xlim(0,17.5)
# plt.xticks([0,2.5,5.0,7.5,10.0,12.5,15.0,17.5])
# plt.ylabel("Number of occurences (n)")
# plt.title("Wind speed distribution (coupled cases)")
# plt.show()

# # Wind rose (coupled)
# H,dir_edges,speed_edges = np.histogram2d(coupled_wd,coupled_ws,bins=[dir_bins,speed_bins])
# freq = H / H.sum()
# bottom = np.zeros(len(rose_theta))
# fig = plt.figure(figsize=(6,6))
# ax = plt.subplot(111,polar=True)
# colors = plt.cm.viridis(np.linspace(0,1,len(speed_bins)-1))
# for i in range(len(speed_bins)-1):
#     values = freq[:, i]
#     bars = ax.bar(rose_theta, values, width=rose_width, bottom=bottom, color=colors[i], label=f"{speed_bins[i]}-{speed_bins[i+1]} m/s")
#     bottom += values
# ax.set_theta_zero_location("N")
# ax.set_theta_direction(-1)
# # ax.set_title("Wind rose (Coupled)")
# ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
# plt.show()

# # Wind rose (coupled) for each time of day
# coupledTimes = coupled_times.dt.hour + coupled_times.dt.minute/60
# night = (coupledTimes > 1) & (coupledTimes < 7)
# sunrising = (coupledTimes >= 7) & (coupledTimes <= 13)
# day = (coupledTimes > 13) & (coupledTimes < 19)
# sunsetting = (coupledTimes >= 19) | (coupledTimes <= 1)

# coupled_ws = wind_speed.sel(time=coupled_times)
# coupled_wd = np.rad2deg(wind_direction.sel(time=coupled_times))

# nightWS = coupled_ws.where(night,drop=True).values.flatten()
# nightWD = coupled_wd.where(night,drop=True).values.flatten()
# sunriseWS = coupled_ws.where(sunrising,drop=True).values.flatten()
# sunriseWD = coupled_wd.where(sunrising,drop=True).values.flatten()
# dayWS = coupled_ws.where(day,drop=True).values.flatten()
# dayWD = coupled_wd.where(day,drop=True).values.flatten()
# sunsetWS = coupled_ws.where(sunsetting,drop=True).values.flatten()
# sunsetWD = coupled_wd.where(sunsetting,drop=True).values.flatten()

# H,dir_edges,speed_edges = np.histogram2d(nightWD,nightWS,bins=[dir_bins,speed_bins])
# freq = H / H.sum()
# bottom = np.zeros(len(rose_theta))
# fig = plt.figure(figsize=(6,6))
# ax = plt.subplot(111,polar=True)
# colors = plt.cm.viridis(np.linspace(0,1,len(speed_bins)-1))
# for i in range(len(speed_bins)-1):
#     values = freq[:, i]
#     bars = ax.bar(rose_theta, values, width=rose_width, bottom=bottom, color=colors[i], label=f"{speed_bins[i]}-{speed_bins[i+1]} m/s")
#     bottom += values
# ax.set_theta_zero_location("N")
# ax.set_theta_direction(-1)
# ax.set_title("Wind rose from 0100-0700 UTC (Coupled cases)")
# ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
# plt.show()

# H,dir_edges,speed_edges = np.histogram2d(sunriseWD,sunriseWS,bins=[dir_bins,speed_bins])
# freq = H / H.sum()
# bottom = np.zeros(len(rose_theta))
# fig = plt.figure(figsize=(6,6))
# ax = plt.subplot(111,polar=True)
# colors = plt.cm.viridis(np.linspace(0,1,len(speed_bins)-1))
# for i in range(len(speed_bins)-1):
#     values = freq[:, i]
#     bars = ax.bar(rose_theta, values, width=rose_width, bottom=bottom, color=colors[i], label=f"{speed_bins[i]}-{speed_bins[i+1]} m/s")
#     bottom += values
# ax.set_theta_zero_location("N")
# ax.set_theta_direction(-1)
# ax.set_title("Wind rose from 0700-1300 UTC (Coupled cases)")
# ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
# plt.show()

# H,dir_edges,speed_edges = np.histogram2d(dayWD,dayWS,bins=[dir_bins,speed_bins])
# freq = H / H.sum()
# bottom = np.zeros(len(rose_theta))
# fig = plt.figure(figsize=(6,6))
# ax = plt.subplot(111,polar=True)
# colors = plt.cm.viridis(np.linspace(0,1,len(speed_bins)-1))
# for i in range(len(speed_bins)-1):
#     values = freq[:, i]
#     bars = ax.bar(rose_theta, values, width=rose_width, bottom=bottom, color=colors[i], label=f"{speed_bins[i]}-{speed_bins[i+1]} m/s")
#     bottom += values
# ax.set_theta_zero_location("N")
# ax.set_theta_direction(-1)
# ax.set_title("Wind rose from 1300-1900 UTC (Coupled cases)")
# ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
# plt.show()

# H,dir_edges,speed_edges = np.histogram2d(sunsetWD,sunsetWS,bins=[dir_bins,speed_bins])
# freq = H / H.sum()
# bottom = np.zeros(len(rose_theta))
# fig = plt.figure(figsize=(6,6))
# ax = plt.subplot(111,polar=True)
# colors = plt.cm.viridis(np.linspace(0,1,len(speed_bins)-1))
# for i in range(len(speed_bins)-1):
#     values = freq[:, i]
#     bars = ax.bar(rose_theta, values, width=rose_width, bottom=bottom, color=colors[i], label=f"{speed_bins[i]}-{speed_bins[i+1]} m/s")
#     bottom += values
# ax.set_theta_zero_location("N")
# ax.set_theta_direction(-1)
# ax.set_title("Wind rose from 1900-0100 UTC (Coupled cases)")
# ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
# plt.show()

# plt.hist(nightWS,bins=50)
# plt.xlabel("Wind speed (m/s)")
# plt.xlim(0,17.5)
# plt.xticks([0,2.5,5.0,7.5,10.0,12.5,15.0,17.5])
# plt.ylabel("Number of occurences (n)")
# plt.title("Wind speed distribution 0100-0700 UTC (coupled cases)")
# plt.show()

#%% Summary statistics for whole summer

# Overall wind rose
all_ws = wind_speed.values.flatten()
all_ws = all_ws[~np.isnan(all_ws)]
all_wd = np.rad2deg(wind_direction.values.flatten())
all_wd = all_wd[~np.isnan(all_wd)]

dir_bins = np.arange(0,361,30)
counts, _ = np.histogram(all_wd,bins=dir_bins)
freq = counts / counts.sum() * 100
rose_theta = np.deg2rad(dir_bins[:-1])
rose_width = np.deg2rad(30)
speed_bins = [0,2,4,6,8,10,12,14,16,18,20,22,24]

H,dir_edges,speed_edges = np.histogram2d(all_wd,all_ws,bins=[dir_bins,speed_bins])
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
# ax.set_title("Wind Rose (Summer 2024)")
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.show()

# # Overall wind speed distribution
# plt.hist(all_ws,bins=50)
# plt.xlabel("Wind speed (m/s)")
# # plt.xlim(0,17.5)
# # plt.xticks([0,2.5,5.0,7.5,10.0,12.5,15.0,17.5])
# plt.ylabel("Number of occurences (n)")
# plt.title("Wind speed distribution (Summer 2024)")
# plt.show()

# # plot dTheta averages for overview
# df_surf = dTheta_surf_mean.to_dataframe(name="dthetasurf_hm").reset_index()
# df_surf["hour"] = df_surf["time"].dt.hour
# df_surf = df_surf.dropna(subset="dthetasurf_hm")
# plt.figure(figsize=(12, 5))
# sb.violinplot(df_surf,x="hour",y="dthetasurf_hm",cut=0,color="lightcoral")
# # define quartiles
# grouped = df_surf.groupby("hour")["dthetasurf_hm"]
# hours = []
# means = []
# q1s, q3s = [], []
# for h, vals in grouped:
#     vals = vals.dropna()
#     hours.append(h)
#     means.append(vals.mean())
#     q1s.append(vals.quantile(0.25))
#     q3s.append(vals.quantile(0.75))
# plt.vlines(hours, q1s, q3s,
#             color="red",
#             linewidth=3,
#             alpha=0.8,
#             label="IQR (25–75%)")
# ax = plt.gca()
# ax.set_xticks([0,2,4,6,8,10,12,14,16,18,20,22,24])
# # ax.axvline(sunrise,color="purple",linestyle="--",linewidth=1.5,label='Sunrise')
# # ax.axvline(sunset,color="black",linestyle="--",linewidth=1.5,label='Sunset')
# ax.axhline(0,color='k')
# ax.set_ylim([-0.05,0.05])
# ax.legend()
# plt.title("Average surface-level dθ/dz throughout the day")
# plt.xlabel("UTC Time")
# plt.ylabel("dθ/dz")
# plt.tight_layout()
# plt.show()

# df_hub = dTheta_hub_mean.to_dataframe(name="dthetahub_hm").reset_index()
# df_hub["hour"] = df_hub["time"].dt.hour
# df_hub = df_hub.dropna(subset="dthetahub_hm")
# plt.figure(figsize=(12, 5))
# sb.violinplot(df_hub,x="hour",y="dthetahub_hm",cut=0,color="lightcoral")
# # define quartiles
# grouped = df_hub.groupby("hour")["dthetahub_hm"]
# hours = []
# means = []
# q1s, q3s = [], []
# for h, vals in grouped:
#     vals = vals.dropna()
#     hours.append(h)
#     means.append(vals.mean())
#     q1s.append(vals.quantile(0.25))
#     q3s.append(vals.quantile(0.75))
# plt.vlines(hours, q1s, q3s,
#             color="red",
#             linewidth=3,
#             alpha=0.8,
#             label="IQR (25–75%)")
# ax = plt.gca()
# ax.set_xticks([0,2,4,6,8,10,12,14,16,18,20,22,24])
# # ax.axvline(sunrise,color="purple",linestyle="--",linewidth=1.5,label='Sunrise')
# # ax.axvline(sunset,color="black",linestyle="--",linewidth=1.5,label='Sunset')
# ax.axhline(0,color='k')
# ax.set_ylim([-0.05,0.05])
# ax.legend()
# plt.title("Average hub-level dθ/dz throughout the day")
# plt.xlabel("UTC Time")
# plt.ylabel("dθ/dz")
# plt.tight_layout()
# plt.show()

