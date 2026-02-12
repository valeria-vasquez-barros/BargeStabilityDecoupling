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
theta = dataAssist["theta"].sel(time=slice("2024-07-15 00:00:00","2024-07-15 23:50:50"))
# theta = dataAssist["theta"]
temp = dataAssist["temperature"].sel(time=slice("2024-07-15 00:00:00","2024-07-15 23:50:50"))
# temp = dataAssist["temperature"]
dTheta = theta.differentiate("height")

# Compare "near-surface" and "hub-height"
dTheta_surf = dTheta.sel(height=slice(40,60))
dTheta_hub = dTheta.sel(height=slice(120,160))
dTheta_times = dTheta.time

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
# print(events)

    # found = False
    
    # if np.any(logic1):
    #     print("statically stable near surface and statically unstable near hub at:",times1.values)
    #     found = True
    # if np.any(logic2):
    #     print("statically unstable near surface and statically stable near hub:",times2.values)
    #     found = True
    # if not found:
    #     print("no decoupling detected")

# collect sunrise/sunset info
location = LocationInfo(latitude=dataAssist.VIP_station_lat, longitude=dataAssist.VIP_station_lon, timezone="UTC")
date = pd.to_datetime(dataAssist.time.values[0])
s=sun(location.observer, date=date)
sunrise = s["sunrise"]
sunset = s["sunset"]

# # plot dTheta along height and time:
# plt.figure(figsize=(10, 5))
# dTheta.plot(x="time", y="height", cmap="coolwarm")
# ax = plt.gca()
# ax.xaxis.set_major_formatter(mdates.DateFormatter("%H"))  # only show hours
# ax.axvline(sunrise,color="purple",linestyle="--",linewidth=1.5,label='Sunrise')
# ax.axvline(sunset,color="black",linestyle="--",linewidth=1.5,label='Sunset')
# ax.legend(loc="upper right")
# plt.title("15 July, 2024")
# plt.xlabel("UTC Time")
# plt.ylabel("Height (m)")
# plt.tight_layout()
# plt.show()

# grab wind speed, wind direction from combined lidar file
# wind_speed = dataLidar["wind_speed"]
wind_speed = dataLidar["wind_speed"].sel(time=slice("2024-07-15 00:10:00","2024-07-16 00:00:00"))
# wind_direction = dataLidar["wind_direction"]
wind_direction = dataLidar["wind_direction"].sel(time=slice("2024-07-15 00:10:00","2024-07-16 00:00:00"))

# calculate u and v
uGeo = -wind_speed * np.sin(wind_direction)
vGeo = -wind_speed * np.cos(wind_direction)
# sGeo = np.sqrt(uGeo**2+vGeo**2)

# sGeo_surf = sGeo.sel(height=slice(40,60))
# sGeo_hub = sGeo.sel(height=slice(120,160))

# # plot wind speed along height and time
# fig,ax = plt.subplots(figsize=(10,5))
# T,Z = np.meshgrid(uGeo["time"],uGeo["height"],indexing="ij")
# q = ax.quiver(T[::5],Z[::5],uGeo[::5],vGeo[::5],sGeo[::5])
# plt.colorbar(q,ax=ax,label="Wind Speed (m/s)")
# ax.xaxis.set_major_formatter(mdates.DateFormatter("%H"))  # only show hours
# ax.set_xlabel("UTC Time")
# ax.set_ylabel("Height (m)")
# ax.set_title("Wind Vector Field [combined lidar file] on 20 July, 2024")
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
deltaTheta_surf = thetasurf_f.values - thetasurf_i.values # K
# 3) change in temperature
tempsurf_i = temp.sel(height=hsurf_i)
tempsurf_f = temp.sel(height=hsurf_f)
dTemp_surf = tempsurf_f.values - tempsurf_i.values
avgTemp_surf = dTemp_surf/dZ_surf + 273.15 # K
# 4) change in u,v over heights
usurf_i = uGeo.sel(height=hsurf_i)
usurf_f = uGeo.sel(height=hsurf_f)
dU_surf = usurf_f.values - usurf_i.values
vsurf_i = vGeo.sel(height=hsurf_i)
vsurf_f = vGeo.sel(height=hsurf_f)
dV_surf = vsurf_f.values - vsurf_i.values
# 5) final calculation
num1_surf = g/avgTemp_surf
num2_surf = deltaTheta_surf*dZ_surf
sGeo_surf = (dU_surf**2+dV_surf**2)
num3_surf = num2_surf/sGeo_surf
BulkRi_surf = num1_surf*num3_surf
# BulkRi_surf = xr.DataArray(bulk_Ri_surf, coords = {"time": dataAssist.time}, dims = ("time"))

# calculate hub Bulk Richardson number:
# 1) establish height difference
hhub_i = 120
hhub_f = 160
dZ_hub = hhub_f-hhub_i

# 2) change in potential temperature
thetahub_i = theta.sel(height=hhub_i)
thetahub_f = theta.sel(height=hhub_f)
deltaTheta_hub = thetahub_f.values - thetahub_i.values # K

# 3) change in temperature
temphub_i = temp.sel(height=hhub_i)
temphub_f = temp.sel(height=hhub_f)
dTemp_hub = temphub_f.values - temphub_i.values
avgTemp_hub = dTemp_hub/dZ_hub + 273.15 # K

# 4) change in u,v over heights
uhub_i = uGeo.sel(height=hhub_i)
uhub_f = uGeo.sel(height=hhub_f)
dU_hub = uhub_f.values - uhub_i.values
vhub_i = vGeo.sel(height=hhub_i)
vhub_f = vGeo.sel(height=hhub_f)
dV_hub = vhub_f.values - vhub_i.values

# 5) final calculation
num1_hub = g/avgTemp_hub
num2_hub = deltaTheta_hub*dZ_hub
sGeo_hub = (dU_hub**2+dV_hub**2)
num3_hub = num2_hub/sGeo_hub
BulkRi_hub = num1_hub*num3_hub
# BulkRi_hub = xr.DataArray(bulk_Ri_hub, coords = {"time": dataAssist.time}, dims = ("time"))

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
# plot Bulk Richardson number over time
# BulkRi.plot()
# ax = plt.gca()
# ax.xaxis.set_major_formatter(mdates.DateFormatter("%H"))  # only show hours
# ax.axhline(0.25, color="orange", linestyle="--", label='Critical Ri')
# ax.axvline(sunrise,color="purple",linestyle="--",linewidth=1.5,label='Sunrise')
# ax.axvline(sunset,color="black",linestyle="--",linewidth=1.5,label='Sunset')
# ax.legend(loc="lower right")
# plt.title("20 July, 2024")
# plt.xlabel("UTC Time")
# plt.ylabel("Bulk Richardson number between 60-200 m")
# plt.tight_layout()
# plt.show()

# # calculate Bulk Richardson number:
# # 1) establish height difference
# g = 9.81
# h_i = 60
# h_f = 200
# deltaZ = h_f-h_i

# # 2) change in potential temperature
# theta_i = theta.sel(height=h_i)
# theta_f = theta.sel(height=h_f)
# deltaTheta = theta_f.values - theta_i.values # K

# # 3) change in temperature
# temp_i = temp.sel(height=h_i)
# temp_f = temp.sel(height=h_f)
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
# BulkRi = xr.DataArray(bulk_Ri, coords = {"time": dataAssist.time}, dims = ("time"))

# # plot Bulk Richardson number over time
# BulkRi.plot()
# ax = plt.gca()
# ax.xaxis.set_major_formatter(mdates.DateFormatter("%H"))  # only show hours
# ax.axhline(0.25, color="orange", linestyle="--", label='Critical Ri')
# ax.axvline(sunrise,color="purple",linestyle="--",linewidth=1.5,label='Sunrise')
# ax.axvline(sunset,color="black",linestyle="--",linewidth=1.5,label='Sunset')
# ax.legend(loc="lower right")
# plt.title("20 July, 2024")
# plt.xlabel("UTC Time")
# plt.ylabel("Bulk Richardson number between 60-200 m")
# plt.tight_layout()
# plt.show()