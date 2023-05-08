import numpy as np
import scipy.io
import os
import matplotlib.pyplot as plt
import pyproj

# charger le fichier .mat
data = scipy.io.loadmat('../../data/GDEM/Jun_GDEMV.mat')
map = scipy.io.loadmat('../../data/GDEM/geomap_GDEMV.mat')
print(map)

salinity=data['salinity']
temp=data['water_temp']*0.001+15

salinity[salinity==-32000]=np.nan
salinity=salinity*0.001+15
print(salinity)

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
print(lat_cast2)
lon_cast2 = - 68 + (41.58/60)
print(lon_cast2)

wgs84 = pyproj.CRS('EPSG:4326')
latlong = pyproj.CRS('EPSG:4269')
lat_deg, lon_deg = pyproj.transform(wgs84, latlong, lon_cast2, lat_cast2)

lat_idx = np.abs(lat-lat_deg).argmin()
lon_idx = np.abs(lon-lon_deg).argmin()


print(lat_idx)
print(lon_idx)
print(temp[:, lat_idx, lon_idx])
print(salinity)

temp_cast2 = temp[:, lat_idx, lon_idx]
salinity_cast2 = salinity[:, lat_idx, lon_idx]


plt.plot(temp_cast2,depth)
plt.show()
