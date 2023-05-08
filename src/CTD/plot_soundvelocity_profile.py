import matplotlib.pyplot as plt
import numpy as np
from osgeo import gdal
import pyproj
import folium
import sys, os

# # specify the path of the folder that contains the data files
# data_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', ..', 'data/')
# print(data_folder)
#
# # add the data folder to the system path
# sys.path.insert(1, data_folder)

# Data loading
data_file1 = "../../data/AE2008_CTD/AE2008_PostDeployment_Cast1_deriv.cnv"
data_file2 = "../../data/AE2008_CTD/AE2008_PostDeployment_Cast2_deriv.cnv"
data_file4 = "../../data/AE2008_CTD/AE2008_TestCast1_deriv.cnv"
data_file3 = "../../data/AE2008_CTD/HS1382C1_deriv.cnv"
data1 = np.loadtxt(data_file1,skiprows=348)
data2 = np.loadtxt(data_file2,skiprows=348)
data3 = np.loadtxt(data_file3,skiprows=348)
data4 = np.loadtxt(data_file4,skiprows=348)

# Extract Depth and sound velocity profile
depths_cast1 = -data1[:,0]
sv1_cast1 = data1[:,20]
sv2_cast1 = data1[:,21]
sv3_cast1 = data1[:,22]

depths_cast2 = -data2[:,0]
sv1_cast2 = data2[:,20]
sv2_cast2 = data2[:,21]
sv3_cast2 = data2[:,22]

#Comparison of the different profiles

fig, axs = plt.subplots(2, 2, figsize=(16, 10))
plt.suptitle("Sound Velocity profile during Bermuda Campaign from CTD Data")

axs[0, 0].plot(sv1_cast1, depths_cast1, label="Chen-Millero")
# axs[0, 0].plot(sv1_cast2[:np.argmin(depths_cast1)], depths_cast2[:np.argmin(depths_cast1)], label="Chen-Millero")
# # axs[0, 0].plot(sv2_cast1, depths_cast1, label="Delgrosso")
# # axs[0, 0].plot(sv3_cast1, depths_cast1, label="Wilson")
# axs[0,0].plot(depths_cast1,label='cast1')
# axs[0,0].plot(depths_cast2,label='cast2')
axs[0, 0].set_xlabel("Sound Velocity (m/s)")
axs[0, 0].set_ylabel("Depth (m)")
axs[0, 0].set_title("Cast1")
axs[0, 0].legend()



axs[1, 0].plot(sv1_cast2, depths_cast2, label="Chen-Millero")
axs[1, 0].plot(sv2_cast2, depths_cast2, label="Delgrosso")
axs[1, 0].plot(sv3_cast2, depths_cast2, label="Wilson")
axs[1, 0].set_xlabel("Sound Velocity (m/s)")
axs[1, 0].set_ylabel("Depth (m)")
axs[1, 0].set_title("Cast2")
axs[1, 0].legend()

depths_HS1382C1 = -data3[:,0]
sv1_HS1382C1 = data3[:,20]
sv2_HS1382C1 = data3[:,21]
sv3_HS1382C1 = data3[:,22]

axs[1, 1].plot(sv1_HS1382C1, depths_HS1382C1, label="Chen-Millero")
axs[1, 1].plot(sv2_HS1382C1, depths_HS1382C1, label="Delgrosso")
axs[1, 1].plot(sv3_HS1382C1, depths_HS1382C1, label="Wilson")
axs[1, 1].set_xlabel("Sound Velocity (m/s)")
axs[1, 1].set_ylabel("Depth (m)")
axs[1, 1].set_title("HS1382C1")
axs[1, 1].legend()

depths_testcast1 = -data4[:,0]
sv1_testcast1 = data4[:,20]
sv2_testcast1 = data4[:,21]
sv3_testcast1 = data4[:,22]

axs[0, 1].plot(sv1_testcast1, depths_testcast1, label="Chen-Millero")
axs[0, 1].plot(sv2_testcast1, depths_testcast1, label="Delgrosso")
axs[0, 1].plot(sv3_testcast1, depths_testcast1, label="Wilson")
axs[0, 1].set_xlabel("Sound Velocity (m/s)")
axs[0, 1].set_ylabel("Depth (m)")
axs[0, 1].set_title("Test Cast1")
axs[0, 1].legend()

plt.show()

#Comparison of the three sound velocity profiles : Chen-Millero, Delgrosso and Wilson
fig, axs = plt.subplots(3, 1, figsize=(8, 10))

#Chen-Millero
axs[0].plot(sv1_cast2, depths_cast2,'b')
axs[0].set_xlabel("Sound Velocity (m/s)")
axs[0].set_ylabel("Depth (m)")
axs[0].set_title("Cast2 - Chen-Millero")
axs[0].legend()

#Delgrosso
axs[1].plot(sv2_cast2, depths_cast2,'orange')
axs[1].set_xlabel("Sound Velocity (m/s)")
axs[1].set_ylabel("Depth (m)")
axs[1].set_title("Cast2 - Delgrosso")
axs[1].legend()

#Wilson
axs[2].plot(sv3_cast2, depths_cast2,'green')
axs[2].set_xlabel("Sound Velocity (m/s)")
axs[2].set_ylabel("Depth (m)")
axs[2].set_title("Cast2 - Wilson")
axs[2].legend()

plt.tight_layout()
plt.show()


min_index_cast2 = np.argmin(depths_cast2)

#Comparison CTD going up and down
fig, axs = plt.subplots(3, 2, figsize=(12, 14))

#Chen-Millero
axs[0,0].plot(sv1_cast2[:min_index_cast2+1], depths_cast2[:min_index_cast2+1],'b')
axs[0,0].set_xlabel("Sound Velocity (m/s)")
axs[0,0].set_ylabel("Depth (m)")
axs[0,0].set_title("Down - Chen-Millero")

#Delgrosso
axs[1,0].plot(sv2_cast2[:min_index_cast2+1], depths_cast2[:min_index_cast2+1],'orange')
axs[1,0].set_xlabel("Sound Velocity (m/s)")
axs[1,0].set_ylabel("Depth (m)")
axs[1,0].set_title("Down - Delgrosso")

#Wilson
axs[2,0].plot(sv3_cast2[:min_index_cast2+1], depths_cast2[:min_index_cast2+1],'g')
axs[2,0].set_xlabel("Sound Velocity (m/s)")
axs[2,0].set_ylabel("Depth (m)")
axs[2,0].set_title("Down - Wilson")


#Chen-Millero
axs[0,1].plot(sv1_cast2[min_index_cast2+1:], depths_cast2[min_index_cast2+1:],'b')
axs[0,1].set_xlabel("Sound Velocity (m/s)")
axs[0,1].set_ylabel("Depth (m)")
axs[0,1].set_title("Up - Chen-Millero")

#Delgrosso
axs[1,1].plot(sv2_cast2[min_index_cast2+1:], depths_cast2[min_index_cast2+1:],'orange')
axs[1,1].set_xlabel("Sound Velocity (m/s)")
axs[1,1].set_ylabel("Depth (m)")
axs[1,1].set_title("Up - Delgrosso")

#Wilson
axs[2,1].plot(sv3_cast2[min_index_cast2+1:], depths_cast2[min_index_cast2+1:],'g')
axs[2,1].set_xlabel("Sound Velocity (m/s)")
axs[2,1].set_ylabel("Depth (m)")
axs[2,1].set_title("Up - Wilson")

plt.subplots_adjust(hspace=0.5)
plt.suptitle('Cast2 - Up and Down')
plt.show()

#Localisation of the different profiles using Folium

#cast1
lat_cast1 = 31 + (26.90/60)
lon_cast1 = - 68 + (41.84/60)

# cast2
lat_cast2 = 31 + (26.95/60)
lon_cast2 = - 68 + (41.58/60)

# HS1382C1
lat_HS1382C1= 32 + (10.02/60)
lon_HS1382C1 = - 64 + (30.14/60)

# Test cast1
lat_testcast1 = 30 + (38.93/60)
lon_testcast1 = - 77 + (36.38/60)

lat=[lat_cast1,lat_cast2,lat_HS1382C1,lat_testcast1]
lon=[lon_cast1,lon_cast2,lon_HS1382C1,lon_testcast1]
labels = ['Cast1', 'Cast2', 'HS1382C1', 'Testcast1']

#Map creation
m = folium.Map(location=[31.5, -68], zoom_start=8)

#Add marker and label
for lat, lon, label in zip(lat, lon, labels):
    folium.Marker(location=[lat, lon], tooltip=label).add_to(m)

#Save Map and open with a webbrowser
# m.save('map.html')
