import scipy.io as sio
import matplotlib.pyplot as plt

# Charger les fichiers .mat
data1 = sio.loadmat('/home/thalia/Documents/seafloorgeodesy/src/esv/castbermuda/mat_interp/data_5229_step_angle_0_01.mat')
data2 = sio.loadmat('/home/thalia/Documents/seafloorgeodesy/src/esv/GDEM/mat_interp/data_5229_step_angle_0_01.mat')

# Récupérer les données des fichiers
angles1 = data1['elevation angle (deg)'].flatten()
esv1 = data1['esv (m/s)'].flatten()

angles2 = data2['elevation angle (deg)'].flatten()
esv2 = data2['esv (m/s)'].flatten()

difference = esv1 - esv2

# Créer une figure avec 2 subplots côte à côte
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Tracer les données dans le premier subplot
ax1.plot(angles1, esv1, label='Data from Bermuda Cast')
ax1.plot(angles2, esv2, label='Data from GDEM', linestyle='--')
ax1.set_xlabel('Elevation angle (deg)')
ax1.set_ylabel('Effective sound velocity (m/s)')
ax1.set_title('Comparison of effective sound velocity with different SV profiles')
ax1.legend()
ax1.grid(True)

# Tracer la différence dans le second subplot
ax2.plot(angles1, difference, label='Difference between Bermuda Cast and GDEM', color='red')
ax2.set_xlabel('Elevation angle (deg)')
ax2.set_ylabel('Difference (m/s)')
ax2.set_title('Difference between the two datasets')
ax2.grid(True)

# Ajuster la mise en page pour éviter les chevauchements
plt.tight_layout()
plt.show()


# Charger les fichiers .mat
data_orig = sio.loadmat('/home/thalia/Documents/seafloorgeodesy/src/esv/castbermuda/data_5220_step_angle_5.mat')
data_interp = sio.loadmat('/home/thalia/Documents/seafloorgeodesy/src/esv/castbermuda/mat_interp/data_5220_step_angle_0_01.mat')

# Récupérer les données des fichiers
angles_orig = data_orig['list_beta'].flatten()
esv_orig = data_orig['c_esv'].flatten()

angles_interp = data_interp['elevation angle (deg)'].flatten()
esv_interp = data_interp['esv (m/s)'].flatten()

# Tracer les données
plt.figure(figsize=(10, 6))

# Utiliser scatter pour les données originales
plt.scatter(angles_orig, esv_orig, color='red', marker='o', label='Original data (5° step)')
# Utiliser plot pour les données interpolées
plt.plot(angles_interp, esv_interp, color='blue', label='Interpolated data (0.01° step)')

# Définir les étiquettes et le titre
plt.xlabel('Elevation angle (deg)')
plt.ylabel('Effective sound velocity (m/s)')
plt.title('Comparison of original and interpolated data')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
