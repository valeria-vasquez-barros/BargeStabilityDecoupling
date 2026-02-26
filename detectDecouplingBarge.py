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

# Grab theta, temp variables from combined assist file
theta = dataAssist["theta"].sel(time=slice("2024-07-04 00:00:00","2024-07-04 23:50:00"))
temp = dataAssist["temperature"].sel(time=slice("2024-07-04 00:00:00","2024-07-04 23:50:00"))
dTheta = theta.differentiate("height")

# Compare "near-surface" and "hub-height"
dTheta_surf = dTheta.sel(height=slice(40,60))
dTheta_hub = dTheta.sel(height=slice(120,160))
dTheta_times = dTheta.time

# static stability quadrant analysis:
dTheta_surf_mean = dTheta_surf.mean("height")
dTheta_hub_mean = dTheta_hub.mean("height")
Q1 = (dTheta_surf_mean > 0) & (dTheta_hub_mean > 0)
Q2 = (dTheta_surf_mean > 0) & (dTheta_hub_mean < 0)
Q3 = (dTheta_surf_mean < 0) & (dTheta_hub_mean < 0)
Q4 = (dTheta_surf_mean < 0) & (dTheta_hub_mean > 0)

plt.figure(figsize=(6,6))
plt.scatter(dTheta_surf_mean[Q1],dTheta_hub_mean[Q1],color='black',alpha=0.4,label="Coupled Stability")
plt.scatter(dTheta_surf_mean[Q2],dTheta_hub_mean[Q2],color='blue',alpha=0.4,label="Surface Stable - Hub Unstable")
plt.scatter(dTheta_surf_mean[Q3],dTheta_hub_mean[Q3],color='gray',alpha=0.4,label="Coupled Instability")
plt.scatter(dTheta_surf_mean[Q4],dTheta_hub_mean[Q4],color='red',alpha=0.4,label="Surface Unstable - Hub Stable")
plt.axhline(0,color='k')
plt.axvline(0,color='k')
plt.xlabel("Surface (40-60m) Static Stability")
plt.ylabel("Hub (120-160m) Static Stability")
plt.title("Static Stability Quadrant Analysis")
plt.legend()
plt.show()

# static decoupling occurrences:
static_decoupled = Q2 | Q4
staticOverall_percent = 100*static_decoupled.mean()
monthly_percent = 100*(static_decoupled.groupby("time.month").mean())
print(f"{staticOverall_percent.values:.2f}% of the summer is statically decoupled")
    
# collect sunrise/sunset info
location = LocationInfo(latitude=dataAssist.VIP_station_lat, longitude=dataAssist.VIP_station_lon, timezone="UTC")
date = pd.to_datetime(dataAssist.time.values[0])
s=sun(location.observer, date=date)
sunrise = s["sunrise"]
sunset = s["sunset"]

def detect_staticdecoupling(dTheta_surf,dTheta_hub):
    
    surf_stable = (dTheta_surf>0).any(dim="height")
    surf_unstable = (dTheta_surf<0).any(dim="height")
    hub_stable = (dTheta_hub>0).any(dim="height")
    hub_unstable = (dTheta_hub<0).any(dim="height")
    
    logic1 = surf_stable & hub_unstable
    logic2 = surf_unstable & hub_stable
    
    times1 = dTheta_surf.time.where(logic1,drop=True)
    times2 = dTheta_surf.time.where(logic2,drop=True)
    
    return {
        "statically stable near surface and statically unstable near hub:": times1,
        "statically unstable near surface and statically stable near hub:": times2,
        }

events = detect_staticdecoupling(dTheta_surf,dTheta_hub)
print(events)

# # plot dTheta at surface:
# plt.figure(figsize=(10, 5))
# dTheta_surf.plot(x="time", y="height", cmap="coolwarm")
# ax = plt.gca()
# ax.xaxis.set_major_formatter(mdates.DateFormatter("%H"))  # only show hours
# ax.axvline(sunrise,color="purple",linestyle="--",linewidth=1.5,label='Sunrise')
# ax.axvline(sunset,color="black",linestyle="--",linewidth=1.5,label='Sunset')
# ax.legend(loc="upper right")
# plt.title("20 July, 2024 [combined]")
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
# plt.title("20 July, 2024 [combined]")
# plt.xlabel("UTC Time")
# plt.ylabel("Height (m)")
# plt.tight_layout()
# plt.show()

# grab wind speed, wind direction from combined lidar file
wind_speed = dataLidar["wind_speed"].sel(time=slice("2024-07-04 00:00:00","2024-07-04 23:50:00"))
wind_direction = np.deg2rad(dataLidar["wind_direction"]).sel(time=slice("2024-07-04 00:00:00","2024-07-04 23:50:00"))

# calculate u and v
uGeo = -wind_speed * np.sin(wind_direction)
vGeo = -wind_speed * np.cos(wind_direction)
sGeo = np.sqrt(uGeo**2+vGeo**2)

# plot wind:
fig,ax = plt.subplots(figsize=(10,5))
T,Z = np.meshgrid(uGeo["time"],uGeo["height"],indexing="ij")
q = ax.quiver(T[::5],Z[::5],uGeo[::5],vGeo[::5],sGeo[::5])
plt.colorbar(q,ax=ax,label="Wind Speed (m/s)")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H"))  # only show hours
ax.set_xlabel("UTC Time")
ax.set_ylabel("Height (m)")
ax.set_title("20 July, 2024 [combined]")
plt.show()

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
tempsurf_i = temp.sel(height=hsurf_i)
tempsurf_f = temp.sel(height=hsurf_f)
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
temphub_i = temp.sel(height=hhub_i)
temphub_f = temp.sel(height=hhub_f)
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
    
    dsurf_stable = (BulkRi_surf>0.25)
    dsurf_unstable = (BulkRi_surf<0.25)
    dhub_stable = (BulkRi_hub>0.25)
    dhub_unstable = (BulkRi_hub<0.25)
    
    dlogic1 = dsurf_stable & dhub_unstable
    dlogic2 = dsurf_unstable & dhub_stable
    
    dtimes1 = BulkRi_surf.time.where(dlogic1,drop=True)
    dtimes2 = BulkRi_surf.time.where(dlogic2,drop=True)
    
    return {
        "dynamically stable near surface and dynamically unstable near hub:": dtimes1,
        "dynamically unstable near surface and dynamically stable near hub:": dtimes2,
        }

devents = detect_dynamicdecoupling(BulkRi_surf,BulkRi_hub)
print(devents)

# plot surface Bulk Richardson number:
fig, ax = plt.subplots(figsize=(6,5))
BulkRi_surf.plot(ax=ax)
ax.set_xlim(BulkRi_surf.time.min().values,BulkRi_surf.time.max().values)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H"))
ax.axhline(0.25, linestyle="--", label='Critical Ri')
ax.axvline(sunrise, linestyle="--", label='Sunrise')
ax.axvline(sunset, linestyle="--", label='Sunset')
ax.set_title("4 July 2024 (40-60m)")
ax.set_xlabel("UTC Time")
ax.set_ylabel("Bulk Richardson number")
ax.legend()
plt.tight_layout()
plt.show()

# plot hub height Bulk Richardson number:
fig, ax = plt.subplots(figsize=(6,5))
BulkRi_hub.plot(ax=ax)
ax.set_xlim(BulkRi_hub.time.min().values,BulkRi_hub.time.max())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H"))
ax.axhline(0.25, linestyle="--", label='Critical Ri')
ax.axvline(sunrise, linestyle="--", label='Sunrise')
ax.axvline(sunset, linestyle="--", label='Sunset')
ax.set_title("4 July 2024 (120-160m)")
ax.set_xlabel("UTC Time")
ax.set_ylabel("Bulk Richardson number")
ax.legend()
plt.tight_layout()
plt.show()

# dynamic stability quadrant analysis: [IN PROGRESS]
valid = (np.abs(BulkRi_surf<1)) & (np.abs(BulkRi_hub<1))
Q1 = (BulkRi_surf > 0.25) & (BulkRi_hub > 0.25)
Q2 = (BulkRi_surf > 0.25) & (BulkRi_hub < 0.25)
Q3 = (BulkRi_surf < 0.25) & (BulkRi_hub < 0.25)
Q4 = (BulkRi_surf < 0.25) & (BulkRi_hub > 0.25)

plt.figure(figsize=(6,6))
plt.scatter(BulkRi_surf[Q1],BulkRi_hub[Q1],color='black',alpha=0.4,label="Coupled Stability")
plt.scatter(BulkRi_surf[Q2],BulkRi_hub[Q2],color='blue',alpha=0.4,label="Surface Stable - Hub Turbulent")
plt.scatter(BulkRi_surf[Q3],BulkRi_hub[Q3],color='gray',alpha=0.4,label="Coupled Turbulence")
plt.scatter(BulkRi_surf[Q4],BulkRi_hub[Q4],color='red',alpha=0.4,label="Surface Turbulent - Hub Stable")
plt.axhline(0.25,color='k')
plt.axvline(0.25,color='k')
plt.xlabel("Surface (40-60m) Bulk Richardson Number")
plt.ylabel("Hub (120-160m) Bulk Richardson Number")
plt.title("Dynamic Stability Quadrant Analysis")
plt.legend()
plt.show()

# decoupling percentages:
decoupled = Q2 | Q4
overall_percent = 100*decoupled.mean()
monthly_percent = 100*(decoupled.groupby("time.month").mean())

# frequency along time:
months = xr.DataArray(["May", "Jun", "Jul", "Aug", "Sep"])
plt.figure(figsize=(8,5))
plt.bar(months,monthly_percent.values,width=0.8)
plt.ylim((0,100))
plt.xlabel("UTC Time")
plt.ylabel("Frequency (%)")
plt.title("Dynamic Decoupling Monthly Frequency (Summer 2024)")
plt.show()

print(f"{overall_percent.values:.2f}% of the summer is dynamically decoupled")

# # identify long durations (1+ hours):
# decouple_flag = decoupled.astype(int)
# groups = (decouple_flag.diff("time") != 0).cumsum("time")
# for num in np.unique(groups):
#     segment = decouple_flag.where(groups==num,drop=True)
#     if segment.mean() == 1:
#         duration = len(segment)
#         if duration >= 18:
#             print(segment.time.values[0], "to", segment.time.values[-1])




