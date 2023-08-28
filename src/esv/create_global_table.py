#!/usr/bin/python3

import os,sys
import re
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from tqdm import tqdm
import matplotlib.lines as mlines
from scipy.io import savemat

dossier = 'esv_table_without_tol/mat_interp'

fichiers = [f for f in os.listdir(dossier) if f.endswith('.mat')]
numbers = [int(re.search(r"data_(\d+)_", f).group(1)) for f in fichiers]

sorted_indices = np.argsort(numbers)
numbers = [numbers[i] for i in sorted_indices]
fichiers = [fichiers[i] for i in sorted_indices]

T_glob = []
x_axis = []

# Interpolation
print("Beggining of interpolation")
for i in tqdm(range(len(fichiers) - 1)):
    fichier = fichiers[i]
    chemin_complet = os.path.join(dossier, fichier)
    data = scipy.io.loadmat(chemin_complet)
    mat_data = data['esv (m/s)']
    if len(mat_data.shape) == 1:
        mat_data = mat_data.reshape(1, -1)
    T_glob.append(mat_data[0].tolist())
    x_axis.append(numbers[i])

    if numbers[i+1] - numbers[i] > 1:
        next_file = fichiers[i+1]
        next_data = scipy.io.loadmat(os.path.join(dossier, next_file))['esv (m/s)']

        if len(next_data.shape) == 1:
            next_data = next_data.reshape(1, -1)

        interpolator = interp1d([numbers[i], numbers[i+1]], np.vstack([mat_data, next_data]), axis=0)

        for j in range(numbers[i] + 1, numbers[i+1]):
            interpolated_data = interpolator(j)
            T_glob.append(interpolated_data.tolist())
            x_axis.append(j)

# Extrapolation
print("Beggining of extrapolation")
if x_axis[-1] < 5380:
    for j in tqdm(range(x_axis[-1] + 1, 5381)):  # 5381 pour inclure 5380
        extrapolated_data = []
        for idx in range(len(T_glob[-1])):
            # Utilisation des deux derniers points pour créer un interpolateur linéaire
            interpolator = interp1d([x_axis[-2], x_axis[-1]],
                                    [T_glob[-2][idx], T_glob[-1][idx]],
                                    fill_value="extrapolate", kind='linear')
            extrapolated_data.append(interpolator(j))
        T_glob.append(extrapolated_data)
        x_axis.append(j)



last_file = fichiers[-1]
last_data = scipy.io.loadmat(os.path.join(dossier, last_file))['esv (m/s)']
if len(last_data.shape) == 1:
    last_data = last_data.reshape(1, -1)
T_glob.append(last_data[0])
x_axis.append(numbers[-1])
x_values = np.linspace(45, 90, len(T_glob[0]))

# Creation of dictionary
print("Data saved")
data_to_save = {
    'angle': x_values,
    'distance': np.array(x_axis),
    'matrice': np.array(T_glob)
}

# Sauvegarde des données dans un fichier .mat
output_file_path = os.path.join(dossier, '../global_table_interp/global_table_esv.mat')
savemat(output_file_path, data_to_save)

print(f"Les données ont été sauvegardées dans {output_file_path}")

if __name__ == '__main__':

    # Plot data
    original_indices = [i for i in range(len(T_glob)) if x_axis[i] in numbers]
    interpolated_indices = [i for i in range(len(T_glob)) if x_axis[i] not in numbers and x_axis[i] < max(numbers)]
    extrapolated_indices = [i for i in range(len(T_glob)) if x_axis[i] not in numbers and x_axis[i] > max(numbers)]

    plt.figure(figsize=(12, 6))

    # Plot données originales
    for idx, i in enumerate(original_indices):
        if idx == 0:
            plt.plot(x_values, T_glob[i], label='ESV computed', color='blue')
        else:
            plt.plot(x_values, T_glob[i], color='blue')

    # Plot données interpolées
    for idx, i in enumerate(interpolated_indices):
        if idx == 0:
            plt.plot(x_values, T_glob[i], label='ESV interpolated', color='red')
        else:
            plt.plot(x_values, T_glob[i], color='red')

    # Plot données interpolées
    for idx, i in enumerate(extrapolated_indices):
        if idx == 0:
            plt.plot(x_values, T_glob[i], label='ESV extrapolated', color='green')
        else:
            plt.plot(x_values, T_glob[i], color='green')

    plt.legend()
    plt.title("ESV v/s Elevation angle")
    plt.xlabel("Elevation angle")
    plt.ylabel("Effective sound velocity")
    plt.grid(True)

    # Sauvegarder le schéma en image
    img_output_path = os.path.join(dossier, '../global_table_interp/visualisation_donnees.png')
    img_directory = os.path.dirname(img_output_path)
    plt.savefig(img_output_path, dpi=500)

    plt.show()
