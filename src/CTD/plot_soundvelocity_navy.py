import numpy as np
import scipy.io
import os
import matplotlib.pyplot as plt
import pyproj

# charger le fichier .mat
data = scipy.io.loadmat('GDEM/Jun_GDEMV.mat')
map = scipy.io.loadmat('GDEM/geomap_GDEMV.mat')
print(data)

salinity=data['salinity']*0.001+15
temp=data['water_temp']*0.001+15

print(salinity-temp)
print(temp)

lat=map['lat']
lon=map['lon']
depth=map['depth']

print(salinity.shape)
print(lon.shape)
print(lat.shape)
print(depth.shape)

# cast2
lat_cast2 = 31 + (26.95/60)
lon_cast2 = - 68 + (41.58/60)
lat_deg, lon_deg = pyproj.transform(wgs84, epsg3035, lon_cast2, lat_cast2)

lat_idx = np.abs(lat-lat_cast2).argmin()
lon_idx = np.abs(lon-lon_cast2).argmin()

print(lat_idx)
print(lon_idx)
print(lon+lon_cast2)

print(lat_idx)
print(lon_idx)
print(temp[:, lat_idx, lon_idx])
print(salinity)

temp_cast2 = temp[:, lat_idx, lon_idx]
salinity_cast2 = salinity[:, lat_idx, lon_idx]


plt.plot(salinity_cast2,-depth)
plt.show()
