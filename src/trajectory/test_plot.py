import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

# Supposons que vous ayez chargé la matrice 'data' à partir du fichier MATLAB en utilisant la fonction loadmat
data = sio.loadmat('../../data/SwiftNav_Data/Unit1-camp.mat')
print(data)
#
# # Récupérer les coordonnées XYZ à partir de la matrice 'data'
# x = data['d']['xyz'][0][0][:, 0]
# y = data['d']['xyz'][0][0][:, 1]
# z = data['d']['xyz'][0][0][:, 2]
#
# # Créer le graphique
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# # Tracer les coordonnées XYZ
# ax.scatter(x, y, z, c='b', marker='o')
#
# # Ajouter des étiquettes aux axes
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
#
# # Afficher le graphique
# plt.show()

data = sio.loadmat('../../data/DOG/DOG1/DOG1-camp.mat')

data = data["tags"].astype(float)
print(data)
plt.scatter(data[:,0]/3600,np.unwrap(data[:,1]/1e9*2*np.pi)/(2*np.pi), s = 1)
plt.show()


import datetime

# Exemple : votre temps en secondes
time_in_seconds = 85042.00

# Convertir le temps en une date et heure
reference_datetime = datetime.datetime(2020, 6, 19)  # Remplacez par la référence temporelle appropriée
time_as_datetime = reference_datetime + datetime.timedelta(seconds=time_in_seconds)
print(time_as_datetime)
