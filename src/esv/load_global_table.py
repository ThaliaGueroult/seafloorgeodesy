#!/usr/bin/python3

from scipy.io import loadmat
import os

# Charger les données depuis le fichier .mat
dossier = 'esv_table_without_tol/global_table_interp'
output_file_path = os.path.join(dossier, 'global_table_esv.mat')
loaded_data = loadmat(output_file_path)

# Afficher le contenu
print("Angle:")
print(loaded_data['angle'])

print("\nDistance:")
print(loaded_data['distance'])

print("\nMatrice:")
print(loaded_data['matrice'])


import numpy as np
import scipy.io as sio

# Charger les données depuis le fichier .prd
data_prd = np.genfromtxt('../../data/SwiftNav_Data/Unit4-camp.prd')
days = data_prd[:,0].flatten()
times = data_prd[:,1].flatten()
lat = data_prd[:,5].flatten()
lon = data_prd[:,6].flatten()
elev = data_prd[:,7].flatten()
x = data_prd[:,2].flatten()
y = data_prd[:,3].flatten()
z = data_prd[:,4].flatten()

# Enregistrer ces données dans un fichier .mat
mat_file_path = '../../data/SwiftNav_Data/Unit4-camp_bis.mat'
sio.savemat(mat_file_path, {
    'days': days,
    'times': times,
    'lat': lat,
    'lon': lon,
    'elev': elev,
    'x': x,
    'y': y,
    'z': z
})

print(f"Les données ont été sauvegardées dans {mat_file_path}")
