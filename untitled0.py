# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 12:16:10 2026

@author: valer
"""

import numpy as np
import matplotlib.pyplot as plt

assist_old = np.array([46, 61, 77, 95, 114, 136, 159, 185, 214, 245, 280])
assist_new = np.array([40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300])
lidar = assist_new

# plt.hlines(y=assist_old,xmin=0.5,xmax=1.5,colors='r')
# # plt.hlines(y=lidar,xmin=2,xmax=3)
# plt.xlim(0,2)
# plt.xticks([])
# plt.yticks([50,100,150,200,250,300])
# # plt.legend(['ASSIST','WindCube V2'])
# plt.ylabel('Height (m)')
# # plt.title('ASSIST Spatial Resolution')
# plt.show()

plt.hlines(y=assist_old,xmin=0.5,xmax=1.5,colors='r')
plt.hlines(y=assist_new,xmin=2,xmax=3,colors='r')
plt.hlines(y=lidar,xmin=3.5,xmax=4.5)
plt.xlim(0,8)
plt.xticks([])
plt.legend(['ASSIST','Interpolated ASSIST','WindCube V2'])
plt.ylabel('Height (m)')
# plt.title('Instrument Spatial Resolution')
plt.show()