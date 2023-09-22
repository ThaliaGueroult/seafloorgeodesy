#!/usr/bin/python3

from scipy.io import loadmat
import os
import matplotlib.pyplot as plt
import numpy as np

# Charger les données depuis le fichier .mat
dossier = 'castbermuda/global_table_interp'
output_file_path = os.path.join(dossier, 'global_table_esv.mat')
loaded_data = loadmat(output_file_path)

# Afficher le contenu
print("Angle:")
print(loaded_data['angle'])

print("\nDistance:")
print(loaded_data['distance'])

print("\nMatrice:")
print(loaded_data['matrice'])


import matplotlib.pyplot as plt
import os
from scipy.io import loadmat

# Charger les données depuis le fichier .mat
dossier = 'GDEM/global_table_interp'
output_file_path = os.path.join(dossier, 'global_table_esv.mat')
loaded_data = loadmat(output_file_path)

# Extraire les données
angle = loaded_data['angle'].squeeze()
distance = loaded_data['distance'].squeeze()
matrice = loaded_data['matrice']

# Créer un colormap pour les différentes distances
colors = plt.cm.rainbow(np.linspace(0, 1, len(distance)))

plt.figure(figsize=(12, 8))

# Pour chaque distance, plotter la ligne correspondante de la matrice
for i, dist in enumerate(distance):
    plt.plot(angle, matrice[i], color=colors[i])

# Convertir les distances à une échelle normalisée
normalized_distances = (distance - distance.min()) / (distance.max() - distance.min())

tick_interval = 5
selected_indices = np.arange(0, len(distance), tick_interval)
selected_ticks = (selected_indices / len(distance)).tolist()
selected_labels = [distance[idx] for idx in selected_indices] # nous excluons le dernier élément pour éviter un dépassement d'indice

cbar = plt.colorbar(plt.cm.ScalarMappable(cmap='rainbow'), ticks=selected_ticks)
cbar.set_label('Vertical Distance (m)')
cbar.set_ticklabels(selected_labels)

plt.xlabel('Angle (°)')
plt.ylabel('Effective sound velocity (m/s)')
plt.title('Effective Sound velocity table look-up for Bermuda cast')

plt.tight_layout()
plt.show()
