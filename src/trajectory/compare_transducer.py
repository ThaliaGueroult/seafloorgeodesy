import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.io as sio

# 1. Charger les deux fichiers .mat.
data_transducer_ecef = sio.loadmat('kabsch_average.mat')
transducer_position = sio.loadmat('transducer_positions.mat')

# 2. Extraire lat, lon, elev et times de chaque fichier.
lat_ecef, lon_ecef, elev_ecef, times_ecef = data_transducer_ecef['lat'].flatten(), data_transducer_ecef['lon'].flatten(), data_transducer_ecef['elev'].flatten(), data_transducer_ecef['times'].flatten()
lat_pos, lon_pos, elev_pos, times_pos = transducer_position['lat'].flatten(), transducer_position['lon'].flatten(), transducer_position['elev'].flatten(), transducer_position['times'].flatten()

# 3. Filtrer les données pour les temps communs entre les deux fichiers.
common_times, idx_ecef, idx_pos = np.intersect1d(times_ecef, times_pos, return_indices=True)
lat_ecef, lon_ecef, elev_ecef = lat_ecef[idx_ecef], lon_ecef[idx_ecef], elev_ecef[idx_ecef]
lat_pos, lon_pos, elev_pos = lat_pos[idx_pos], lon_pos[idx_pos], elev_pos[idx_pos]
#... [Vos importations et chargements de données précédents]

#... [Vos importations et chargements de données précédents]

fig = plt.figure(figsize=(12, 12))
gs = gridspec.GridSpec(3, 2, width_ratios=[1, 1], height_ratios=[1, 1, 1])

# Plots pour les différences de latitude, longitude et élévation
ax1 = plt.subplot(gs[0, 0])
ax1.plot(common_times, lat_ecef - lat_pos, label='Difference in Latitude')
ax1.set_ylabel('Difference in Latitude')
ax1.legend()

ax2 = plt.subplot(gs[1, 0])
ax2.plot(common_times, lon_ecef - lon_pos, label='Difference in Longitude')
ax2.set_ylabel('Difference in Longitude')
ax2.legend()

ax3 = plt.subplot(gs[2, 0])
ax3.plot(common_times, elev_ecef - elev_pos, label='Difference in Elevation')
ax3.set_ylabel('Difference in Elevation')
ax3.legend()

# Scatter plot pour lat, lon avec colorbars
ax4 = plt.subplot(gs[:, 1])
sc1 = ax4.scatter(lon_ecef, lat_ecef, c=elev_ecef, s=5, cmap='Reds', label='Attitude & Lever Arm Corrected')
sc2 = ax4.scatter(lon_pos, lat_pos, c=elev_pos, s=5, cmap='Blues', label='Sphere Intersection')
ax4.set_xlabel('Longitude')
ax4.set_ylabel('Latitude')
ax4.legend()

# Créer une position d'axe supplémentaire pour les colorbars
cax1 = fig.add_axes([0.85, 0.1, 0.02, 0.8])
cax2 = fig.add_axes([0.89, 0.1, 0.02, 0.8])

cbar1 = fig.colorbar(sc1, cax=cax1, orientation='vertical')
cbar1.set_label('Elevation (Attitude & Lever Arm Corrected)')

cbar2 = fig.colorbar(sc2, cax=cax2, orientation='vertical')
cbar2.set_label('Elevation (Sphere Intersection)')

plt.tight_layout()
plt.subplots_adjust(right=0.82)
fig.suptitle('Comparison between Transducer Positions Obtained by Two Methods', fontsize=16, y=1.02)
plt.show()


import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# [Vos importations et chargements de données précédents]

# Extraire x, y, z et times de chaque fichier.
x_ecef, y_ecef, z_ecef = data_transducer_ecef['x'].flatten(), data_transducer_ecef['y'].flatten(), data_transducer_ecef['z'].flatten()
x_pos, y_pos, z_pos = transducer_position['x'].flatten(), transducer_position['y'].flatten(), transducer_position['z'].flatten()

# Filtrer les données pour les temps communs entre les deux fichiers.
common_times, idx_ecef, idx_pos = np.intersect1d(times_ecef, times_pos, return_indices=True)
x_ecef, y_ecef, z_ecef = x_ecef[idx_ecef], y_ecef[idx_ecef], z_ecef[idx_ecef]
x_pos, y_pos, z_pos = x_pos[idx_pos], y_pos[idx_pos], z_pos[idx_pos]

# Calculer les différences et la distance
diff_x = x_ecef - x_pos
diff_y = y_ecef - y_pos
diff_z = z_ecef - z_pos
distance = np.sqrt(diff_x**2 + diff_y**2 + diff_z**2)

# Créer les plots
fig, axs = plt.subplots(2, 2, figsize=(15, 10))

# Plot des différences x, y, z et de la distance
axs[0, 0].plot(common_times, diff_x, label='Difference in X')
axs[0, 0].set_ylabel('Difference in X')
axs[0, 0].legend()

axs[1, 0].plot(common_times, diff_y, label='Difference in Y')
axs[1, 0].set_ylabel('Difference in Y')
axs[1, 0].legend()

axs[0, 1].plot(common_times, diff_z, label='Difference in Z')
axs[0, 1].set_ylabel('Difference in Z')
axs[0, 1].legend()

axs[1, 1].plot(common_times, distance, label='Euclidean Distance')
axs[1, 1].set_xlabel('Time')
axs[1, 1].set_ylabel('Distance')
axs[1, 1].legend()

plt.tight_layout()
plt.show()
