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
testAssist = r"C:\Users\valer\Documents\WFIP3\barg.assist.tropoe.z01.c1\barg.assist.tropoe.z01.c1.20240715.000005.nc"
testLidar = r"C:\Users\valer\Documents\WFIP3\barg.lidar.z02.a0\downloader\barg.lidar.z02.a0.20240715.001000.sta.nc"
dataAssist = xr.open_dataset(filepathAssist,decode_times = "true")
dataLidar = xr.open_dataset(filepathLidar,decode_times="true")
dataControlAssist = xr.open_dataset(testAssist,decode_times = "true")
dataControlLidar = xr.open_dataset(testLidar,decode_times = "true")
dataControlAssist = dataControlAssist.assign_coords(height = dataControlAssist["height"] * 1000)
dataControlAssist["time"]=dataControlAssist["time"].dt.floor("10min")

# Grab theta, temp variables from combined assist file
theta = dataAssist["theta"].sel(time=slice("2024-07-15 00:10:00","2024-07-15 23:50:50"))
temp = dataAssist["temperature"].sel(time=slice("2024-07-15 00:10:00","2024-07-15 23:50:50"))
dTheta = theta.differentiate("height")

thetaC = dataControlAssist["theta"].sel(time=slice("2024-07-15 00:10:00","2024-07-15 23:50:50"))
tempC = dataControlAssist["temperature"].sel(time=slice("2024-07-15 00:10:00","2024-07-15 23:50:50"))
thetaExt = thetaC.interp(height = np.linspace(40,300,14),kwargs={"fill_value":"extrapolate"})
tempExt = tempC.interp(height = np.linspace(40,300,14),kwargs={"fill_value":"extrapolate"})
dThetaC = thetaExt.differentiate("height")

# Compare "near-surface" and "hub-height"
dTheta_surf = dTheta.sel(height=slice(40,60))
dTheta_hub = dTheta.sel(height=slice(120,160))
dTheta_times = dTheta.time

dThetaC_surf = dThetaC.sel(height=slice(40,60))
dThetaC_hub = dThetaC.sel(height=slice(120,160))
dThetaC_times = dThetaC.time

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
# eventsC = detect_staticdecoupling(dThetaC_surf,dThetaC_hub)

# print(np.array_equal(
#     events["statically stable near surface and statically unstable near hub:"],
#     eventsC["statically stable near surface and statically unstable near hub:"]
# ))
# print(np.array_equal(
#     events["statically unstable near surface and statically stable near hub:"],
#     eventsC["statically unstable near surface and statically stable near hub:"]
# ))

# plot dTheta along height and time:
plt.figure(figsize=(10, 5))
dTheta_surf.plot(x="time", y="height", cmap="coolwarm")
ax = plt.gca()
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H"))  # only show hours
ax.axvline(sunrise,color="purple",linestyle="--",linewidth=1.5,label='Sunrise')
ax.axvline(sunset,color="black",linestyle="--",linewidth=1.5,label='Sunset')
ax.legend(loc="upper right")
plt.title("15 July, 2024 [combined]")
plt.xlabel("UTC Time")
plt.ylabel("Height (m)")
plt.tight_layout()
plt.show()

# plot dTheta along height and time:
plt.figure(figsize=(10, 5))
dThetaC_surf.plot(x="time", y="height", cmap="coolwarm")
ax = plt.gca()
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H"))  # only show hours
ax.axvline(sunrise,color="purple",linestyle="--",linewidth=1.5,label='Sunrise')
ax.axvline(sunset,color="black",linestyle="--",linewidth=1.5,label='Sunset')
ax.legend(loc="upper right")
plt.title("15 July, 2024 [individual]")
plt.xlabel("UTC Time")
plt.ylabel("Height (m)")
plt.tight_layout()
plt.show()

# grab wind speed, wind direction from combined lidar file
wind_speed = dataLidar["wind_speed"].sel(time=slice("2024-07-15 00:10:00","2024-07-15 23:50:50"))
wind_direction = np.deg2rad(dataLidar["wind_direction"].sel(time=slice("2024-07-15 00:10:00","2024-07-15 23:50:50")))

wind_speedC = dataControlLidar["wind_speed"].sel(time=slice("2024-07-15 00:10:00","2024-07-15 23:50:50"))
wind_directionC = np.deg2rad(dataControlLidar["wind_direction"].sel(time=slice("2024-07-15 00:10:00","2024-07-15 23:50:50")))

# calculate u and v
uGeo = -wind_speed * np.sin(wind_direction)
vGeo = -wind_speed * np.cos(wind_direction)
sGeo = np.sqrt(uGeo**2+vGeo**2)

uGeoC = -wind_speedC * np.sin(wind_directionC)
vGeoC = -wind_speedC * np.cos(wind_directionC)
sGeoC = np.sqrt(uGeoC**2+vGeoC**2)

# plot wind speed along height and time
fig,ax = plt.subplots(figsize=(10,5))
T,Z = np.meshgrid(uGeo["time"],uGeo["height"],indexing="ij")
q = ax.quiver(T[::5],Z[::5],uGeo[::5],vGeo[::5],sGeo[::5])
plt.colorbar(q,ax=ax,label="Wind Speed (m/s)")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H"))  # only show hours
ax.set_xlabel("UTC Time")
ax.set_ylabel("Height (m)")
ax.set_title("15 July, 2024 [combined]")
plt.show()

fig,ax = plt.subplots(figsize=(10,5))
T,Z = np.meshgrid(uGeoC["time"],uGeoC["height"],indexing="ij")
q = ax.quiver(T[::5],Z[::5],uGeoC[::5],vGeoC[::5],sGeoC[::5])
plt.colorbar(q,ax=ax,label="Wind Speed (m/s)")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H"))  # only show hours
ax.set_xlabel("UTC Time")
ax.set_ylabel("Height (m)")
ax.set_title("15 July, 2024 [individual]")
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
# dTemp_surf = tempsurf_f - tempsurf_i
# avgTemp_surf = dTemp_surf/dZ_surf + 273.15 # K, apparently this could be a lapse rate?
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

# 2) change in potential temperature
thetaCsurf_i = thetaExt.sel(height=hsurf_i)
thetaCsurf_f = thetaExt.sel(height=hsurf_f)
deltaThetaC_surf = thetaCsurf_f - thetaCsurf_i # K
# 3) change in temperature
tempCsurf_i = tempExt.sel(height=hsurf_i)
tempCsurf_f = tempExt.sel(height=hsurf_f)
# dTempC_surf = tempsurf_f - tempsurf_i
# avgTempC_surf = dTemp_surf/dZ_surf + 273.15 # K, apparently this could be a lapse rate?
avgTempC_surf = (tempCsurf_i+tempCsurf_f)/2 # K

# 4) change in u,v over heights
uCsurf_i = uGeoC.sel(height=hsurf_i)
uCsurf_f = uGeoC.sel(height=hsurf_f)
dUC_surf = uCsurf_f - uCsurf_i
vCsurf_i = vGeoC.sel(height=hsurf_i)
vCsurf_f = vGeoC.sel(height=hsurf_f)
dVC_surf = vCsurf_f - vCsurf_i
# 5) final calculation
num1C_surf = g/avgTempC_surf
num2C_surf = deltaThetaC_surf*dZ_surf
sGeoC_surf = (dUC_surf**2+dVC_surf**2)
num3C_surf = num2C_surf/sGeoC_surf
BulkRiC_surf = num1C_surf*num3C_surf

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
dTemp_hub = temphub_f - temphub_i
# avgTemp_hub = dTemp_hub/dZ_hub + 273.15 # K, apparently this could be a lapse rate?
avgTemp_hub = (tempsurf_i+tempsurf_f)/2 # K

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
        "dynamically stable near surface and statically unstable near hub:": dtimes1,
        "dynamically unstable near surface and statically stable near hub:": dtimes2,
        }

devents = detect_dynamicdecoupling(BulkRi_surf,BulkRi_hub)
print(devents)

# plot surface Bulk Richardson number
fig, ax = plt.subplots(figsize=(6,5))
BulkRi_surf.plot(ax=ax)
ax.set_xlim(BulkRi_surf.time.min().values,BulkRi_surf.time.max().values)
# ax.set_ylim(-1, 1)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H"))
ax.axhline(0.25, linestyle="--", label='Critical Ri')
# ax.axvline(sunrise, linestyle="--", label='Sunrise')
# ax.axvline(sunset, linestyle="--", label='Sunset')
ax.set_title("15 July 2024 (40-60m) [combined]")
ax.set_xlabel("UTC Time")
ax.set_ylabel("Bulk Richardson number")
ax.legend()
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(6,5))
BulkRiC_surf.plot(ax=ax)
ax.set_xlim(BulkRi_surf.time.min().values,BulkRi_surf.time.max().values)
# ax.set_ylim(-1, 1)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H"))
ax.axhline(0.25, linestyle="--", label='Critical Ri')
# ax.axvline(sunrise, linestyle="--", label='Sunrise')
# ax.axvline(sunset, linestyle="--", label='Sunset')
ax.set_title("15 July 2024 (40-60m) [individual]")
ax.set_xlabel("UTC Time")
ax.set_ylabel("Bulk Richardson number")
ax.legend()
plt.tight_layout()
plt.show()

# plot hub height Bulk Richardson number
fig, ax = plt.subplots(figsize=(5,4))
BulkRi_hub.plot(ax=ax)

ax.set_xlim(BulkRi_hub.time.min().values,BulkRi_hub.time.max())
ax.set_ylim(-1, 1)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H"))
ax.axhline(0.25, linestyle="--", label='Critical Ri')
# ax.axvline(sunrise, linestyle="--", label='Sunrise')
# ax.axvline(sunset, linestyle="--", label='Sunset')

ax.set_title("20 July 2024 (120-160m)")
ax.set_xlabel("UTC Time")
ax.set_ylabel("Bulk Richardson number")
ax.legend()
plt.tight_layout()
plt.show()

# BulkRi_surf.plot(x="time")
# ax = plt.gca()
# ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))  # only show hours
# ax.axhline(0.25, color="orange", linestyle="--", label='Critical Ri')
# ax.axvline(sunrise,color="purple",linestyle="--",linewidth=1.5,label='Sunrise')
# ax.axvline(sunset,color="black",linestyle="--",linewidth=1.5,label='Sunset')
# ax.legend(loc="lower right")
# plt.title("20 July, 2024")
# plt.xlabel("UTC Time")
# plt.ylabel("Bulk Richardson number between 40-60 m")
# plt.tight_layout()
# plt.show()

# BulkRi_hub.plot(x="time")
# ax = plt.gca()
# ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))  # only show hours
# ax.axhline(0.25, color="orange", linestyle="--", label='Critical Ri')
# ax.axvline(sunrise,color="purple",linestyle="--",linewidth=1.5,label='Sunrise')
# ax.axvline(sunset,color="black",linestyle="--",linewidth=1.5,label='Sunset')
# ax.legend(loc="lower right")
# plt.title("20 July, 2024")
# plt.xlabel("UTC Time")
# plt.ylabel("Bulk Richardson number between 120-160 m")
# plt.tight_layout()
# plt.show()
