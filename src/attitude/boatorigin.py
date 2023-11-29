import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D  # Pour la légende


# Charger les données
data_unit1 = sio.loadmat('../../data/SwiftNav_Data/GPStime_synchro/Unit1-camp_bis_GPStime_synchro.mat')
data_unit2 = sio.loadmat('../../data/SwiftNav_Data/GPStime_synchro/Unit2-camp_bis_GPStime_synchro.mat')
data_unit3 = sio.loadmat('../../data/SwiftNav_Data/GPStime_synchro/Unit3-camp_bis_GPStime_synchro.mat')
data_unit4 = sio.loadmat('../../data/SwiftNav_Data/GPStime_synchro/Unit4-camp_bis_GPStime_synchro.mat')

# Extraire et regrouper les coordonnées
coords_unit1 = np.array([data_unit1['x'].flatten(), data_unit1['y'].flatten(), data_unit1['z'].flatten()]).T
coords_unit2 = np.array([data_unit2['x'].flatten(), data_unit2['y'].flatten(), data_unit2['z'].flatten()]).T
coords_unit3 = np.array([data_unit3['x'].flatten(), data_unit3['y'].flatten(), data_unit3['z'].flatten()]).T
coords_unit4 = np.array([data_unit4['x'].flatten(), data_unit4['y'].flatten(), data_unit4['z'].flatten()]).T

# Trouver les indices où il y a des 'nan' pour chaque unité
invalid_indices_unit1 = np.isnan(coords_unit1).any(axis=1)
invalid_indices_unit2 = np.isnan(coords_unit2).any(axis=1)
invalid_indices_unit3 = np.isnan(coords_unit3).any(axis=1)
invalid_indices_unit4 = np.isnan(coords_unit4).any(axis=1)

# Combine les masques pour obtenir un masque unique
combined_invalid_indices = invalid_indices_unit1 | invalid_indices_unit2 | invalid_indices_unit3 | invalid_indices_unit4

# Utilise le masque inverse (i.e., les indices valides) pour filtrer les coordonnées
valid_indices = ~combined_invalid_indices

coords_unit1 = coords_unit1[valid_indices]
coords_unit2 = coords_unit2[valid_indices]
coords_unit3 = coords_unit3[valid_indices]
coords_unit4 = coords_unit4[valid_indices]


days = data_unit1['days'].flatten() - 59015
times = data_unit1['times'].flatten()
datetimes = ((days * 24 * 3600) + times)/3600

datetime = datetimes[valid_indices]

# Calculer l'origine
O = (coords_unit1 + coords_unit2 + coords_unit3 + coords_unit4) / 4

# Calculer l'axe X
X_direction = ((coords_unit3 + coords_unit4) / 2) - ((coords_unit1 + coords_unit2) / 2)
X_direction /= np.linalg.norm(X_direction)  # Normaliser

# Calculer l'axe Y
Y_direction = ((coords_unit2 + coords_unit3) / 2) - ((coords_unit1 + coords_unit4) / 2)
Y_direction /= np.linalg.norm(Y_direction)  # Normaliser

# Calculer l'axe Z
Z_direction = np.cross(X_direction, Y_direction)
Z_direction /= np.linalg.norm(Z_direction)  # Normaliser

def compute_rotation_matrix(X, Y, Z):
    e_x = np.array([1, 0, 0])
    e_y = np.array([0, 1, 0])
    e_z = np.array([0, 0, 1])

    R = np.array([
        [np.dot(X, e_x), np.dot(Y, e_x), np.dot(Z, e_x)],
        [np.dot(X, e_y), np.dot(Y, e_y), np.dot(Z, e_y)],
        [np.dot(X, e_z), np.dot(Y, e_z), np.dot(Z, e_z)]
    ])
    return R

# Exemple pour un ensemble de X, Y, Z
R = compute_rotation_matrix(X_direction[0], Y_direction[0], Z_direction[0])

def extract_euler_angles(R):
    theta_x = np.arctan2(R[2,1], R[2,2])
    theta_y = np.arctan2(-R[2,0], np.sqrt(R[2,1]**2 + R[2,2]**2))
    theta_z = np.arctan2(R[1,0], R[0,0])
    return theta_x, theta_y, theta_z

theta_x, theta_y, theta_z = extract_euler_angles(R)
print(theta_x, theta_y, theta_z)

# Initialisation de listes pour stocker les angles
theta_x_list = []
theta_y_list = []
theta_z_list = []

n = 73506
datetime = np.arange(n)
# Parcourir tous les ensembles de données et calculer les angles d'Euler
for i in range(n):
    R = compute_rotation_matrix(X_direction[i], Y_direction[i], Z_direction[i])
    print(R)
    theta_x, theta_y, theta_z = np.degrees(extract_euler_angles(R))

    theta_x_list.append(theta_x)
    theta_y_list.append(theta_y)
    theta_z_list.append(theta_z)

# Affichage

plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.scatter(datetime, theta_x_list, label="Theta_x")
plt.title("Theta_x au fil du temps")
plt.xlabel("Temps")
plt.ylabel("Theta_x (radians)")

plt.subplot(3, 1, 2)
plt.scatter(datetime, theta_y_list, label="Theta_y")
plt.title("Theta_y au fil du temps")
plt.xlabel("Temps")
plt.ylabel("Theta_y (radians)")

plt.subplot(3, 1, 3)
plt.scatter(datetime, theta_z_list, label="Theta_z")
plt.title("Theta_z au fil du temps")
plt.xlabel("Temps")
plt.ylabel("Theta_z (radians)")

plt.tight_layout()
plt.show()

#
#
# def compute_frame(i):
#     O = (coords_unit1[i] + coords_unit2[i] + coords_unit3[i] + coords_unit4[i]) / 4
#     X_direction = ((coords_unit3[i] + coords_unit4[i]) / 2) - ((coords_unit1[i] + coords_unit2[i]) / 2)
#     X_direction /= np.linalg.norm(X_direction)
#
#     Y_direction = ((coords_unit2[i] + coords_unit3[i]) / 2) - ((coords_unit1[i] + coords_unit4[i]) / 2)
#     Y_direction /= np.linalg.norm(Y_direction)
#
#     Z_direction = np.cross(X_direction, Y_direction)
#     Z_direction /= np.linalg.norm(Z_direction)
#
#     return O, X_direction, Y_direction, Z_direction
#
# # Initialiser la figure 3D
# fig = plt.figure(figsize=(10, 10))
# ax = fig.add_subplot(111, projection='3d')
#
# # Choisissez une origine fixe
# fixed_origin = np.array([0, 0, 0])
#
# # Configurer les limites
# buffer = 2
# ax.set_xlim([-buffer, buffer])
# ax.set_ylim([-buffer, buffer])
# ax.set_zlim([-buffer, buffer])
#
# O, X_direction, Y_direction, Z_direction = compute_frame(0)
#
# # Initialiser les flèches pour X, Y, Z
# quiver = ax.quiver(fixed_origin[0], fixed_origin[1], fixed_origin[2],
#                    X_direction[0], X_direction[1], X_direction[2], color='blue')
# quiverY = ax.quiver(fixed_origin[0], fixed_origin[1], fixed_origin[2],
#                     Y_direction[0], Y_direction[1], Y_direction[2], color='green')
# quiverZ = ax.quiver(fixed_origin[0], fixed_origin[1], fixed_origin[2],
#                     Z_direction[0], Z_direction[1], Z_direction[2], color='red')
#
# def update(i):
#     _, X_direction_current, Y_direction_current, Z_direction_current = compute_frame(i)
#
#     # Mise à jour des données de la flèche pour X, Y, Z
#     quiver.set_segments([[[fixed_origin[0], fixed_origin[1], fixed_origin[2]],
#                           X_direction_current + fixed_origin]])
#     quiverY.set_segments([[[fixed_origin[0], fixed_origin[1], fixed_origin[2]],
#                            Y_direction_current + fixed_origin]])
#     quiverZ.set_segments([[[fixed_origin[0], fixed_origin[1], fixed_origin[2]],
#                            Z_direction_current + fixed_origin]])
#
#     return quiver, quiverY, quiverZ
#
# # Créer des lignes pour la légende
# legend_elements = [Line2D([0], [0], color='blue', lw=2, label='X-axis'),
#                    Line2D([0], [0], color='green', lw=2, label='Y-axis'),
#                    Line2D([0], [0], color='red', lw=2, label='Z-axis')]
#
# ax.legend(handles=legend_elements, loc='upper right')
#
# ani = FuncAnimation(fig, update, frames=len(O), blit=False, interval=200)
# plt.show()

#
# e_x = np.array([1, 0, 0])
# e_y = np.array([0, 1, 0])
# e_z = np.array([0, 0, 1])
#
# roulis = np.arctan2(np.dot(Y_direction, e_z), np.dot(Z_direction, e_z))
# tangage = np.arctan2(np.dot(X_direction, e_z), np.dot(Z_direction, e_z))
# lacet = np.arctan2(np.dot(Y_direction, e_x), np.dot(X_direction, e_x))
#
# # Convertir les angles en degrés
# roulis_deg = np.degrees(roulis)
# tangage_deg = np.degrees(tangage)
# lacet_deg = np.degrees(lacet)
#
# days = data_unit1['days'].flatten() - 59015
# times = data_unit1['times'].flatten()
# datetimes = ((days * 24 * 3600) + times)/3600
#
# datetime = datetimes[valid_indices]
#
# roulis_list, tangage_list, lacet_list = [], [], []
#
# for i in range(len(datetime)):
#     X_direction = ((coords_unit3[i, :] + coords_unit4[i, :]) / 2) - ((coords_unit1[i, :] + coords_unit2[i, :]) / 2)
#     Y_direction = ((coords_unit2[i, :] + coords_unit3[i, :]) / 2) - ((coords_unit1[i, :] + coords_unit4[i, :]) / 2)
#
#     Z_direction = np.cross(X_direction, Y_direction)
#
#     X_direction /= np.linalg.norm(X_direction)
#     Y_direction /= np.linalg.norm(Y_direction)
#     Z_direction /= np.linalg.norm(Z_direction)
#
#     roulis = np.arctan2(np.dot(Y_direction, e_z), np.dot(Z_direction, e_z))
#     tangage = np.arctan2(np.dot(X_direction, e_z), np.dot(Z_direction, e_z))
#     lacet = np.arctan2(np.dot(Y_direction, e_x), np.dot(X_direction, e_x))
#
#     roulis_list.append(np.degrees(roulis))
#     tangage_list.append(np.degrees(tangage))
#     lacet_list.append(np.degrees(lacet))
#
# # Mise en page des subplots avec GridSpec
# gs = gridspec.GridSpec(3, 2, width_ratios=[1, 2])
# fig = plt.figure(figsize=(18, 12))
#
# ax_boat = plt.subplot(gs[:, 0])
# ax_roulis = plt.subplot(gs[0, 1])
# ax_tangage = plt.subplot(gs[1, 1])
# ax_lacet = plt.subplot(gs[2, 1])
#
# # Configuration du graphique du trajet du bateau
# scatter_boat = ax_boat.scatter([], [], c=[], cmap='viridis', s=5)
# ax_boat.set_title('Position du bateau')
# ax_boat.set_xlabel('x')
# ax_boat.set_ylabel('y')
# ax_boat.set_xlim(min(O[:,0]), max(O[:,0]))
# ax_boat.set_ylim(min(O[:,1]), max(O[:,1]))
#
# # Roulis, tangage, lacet
# scatter_roulis = ax_roulis.scatter([], [], color="r", s=5)
# ax_roulis.set_title('Roulis')
# ax_roulis.set_xlabel('datetime (s)')
# ax_roulis.set_ylabel('Degrés')
# ax_roulis.set_xlim([min(datetime), max(datetime)])
# ax_roulis.set_ylim([min(roulis_list), max(roulis_list)])
#
# scatter_tangage = ax_tangage.scatter([], [], color="g", s=5)
# ax_tangage.set_title('Tangage')
# ax_tangage.set_xlabel('datetime (s)')
# ax_tangage.set_ylabel('Degrés')
# ax_tangage.set_xlim([min(datetime), max(datetime)])
# ax_tangage.set_ylim([min(tangage_list), max(tangage_list)])
#
# scatter_lacet = ax_lacet.scatter([], [], color="b", s=5)
# ax_lacet.set_title('Lacet')
# ax_lacet.set_xlabel('datetime (s)')
# ax_lacet.set_ylabel('Degrés')
# ax_lacet.set_xlim([min(datetime), max(datetime)])
# ax_lacet.set_ylim([min(lacet_list), max(lacet_list)])
#
#
#
# # Définir des listes pour accumuler les données
# x_data, y_data, z_data = [], [], []
# time_data = []
# roulis_data, tangage_data, lacet_data = [], [], []
#
# def init():
#     scatter_boat.set_offsets(np.empty((0, 2)))
#     scatter_roulis.set_offsets(np.empty((0, 2)))
#     scatter_tangage.set_offsets(np.empty((0, 2)))
#     scatter_lacet.set_offsets(np.empty((0, 2)))
#     return scatter_boat, scatter_roulis, scatter_tangage, scatter_lacet
#
# def animate(i):
#     x_data.append(O[i, 0])
#     y_data.append(O[i, 1])
#     z_data.append(O[i, 2])
#
#     time_data.append(datetime[i])
#     roulis_data.append(roulis_list[i])
#     tangage_data.append(tangage_list[i])
#     lacet_data.append(lacet_list[i])
#
#     scatter_boat.set_offsets(np.column_stack([x_data, y_data]))
#     scatter_boat.set_array(np.array(z_data))
#
#     scatter_roulis.set_offsets(np.column_stack([time_data, roulis_data]))
#     scatter_tangage.set_offsets(np.column_stack([time_data, tangage_data]))
#     scatter_lacet.set_offsets(np.column_stack([time_data, lacet_data]))
#
#     return scatter_boat, scatter_roulis, scatter_tangage, scatter_lacet
#
# ani = animation.FuncAnimation(fig, animate, frames=len(datetime), init_func=init, blit=True, interval=1e-34)  # pour 10 fois plus rapide
#
# plt.tight_layout()
# plt.show()

# # Calculer les décalages et appliquer la transformation
# transformed_unit2 = coords_unit2 - coords_unit1
# transformed_unit3 = coords_unit3 - coords_unit1
# transformed_unit4 = coords_unit4 - coords_unit1
#
# # Créer les sous-graphiques
# fig, axs = plt.subplots(3, 1, figsize=(10, 15))
#
# # Coordonnées X
# # axs[0].plot(coords_unit1[0], label='Unité 1', marker='o')
# axs[0].plot(transformed_unit2[0], label='Unité 2', marker='x')
# axs[0].plot(transformed_unit3[0], label='Unité 3', marker='s')
# axs[0].plot(transformed_unit4[0], label='Unité 4', marker='d')
# axs[0].set_title('Coordonnées X')
# axs[0].set_xlabel('Index')
# axs[0].set_ylabel('X')
# axs[0].legend()
#
# # Coordonnées Y
# # axs[1].plot(coords_unit1[1], label='Unité 1', marker='o')
# axs[1].plot(transformed_unit2[1], label='Unité 2', marker='x')
# axs[1].plot(transformed_unit3[1], label='Unité 3', marker='s')
# axs[1].plot(transformed_unit4[1], label='Unité 4', marker='d')
# axs[1].set_title('Coordonnées Y')
# axs[1].set_xlabel('Index')
# axs[1].set_ylabel('Y')
# axs[1].legend()
#
# # Coordonnées Z
# # axs[2].plot(coords_unit1[2], label='Unité 1', marker='o')
# axs[2].plot(transformed_unit2[2], label='Unité 2', marker='x')
# axs[2].plot(transformed_unit3[2], label='Unité 3', marker='s')
# axs[2].plot(transformed_unit4[2], label='Unité 4', marker='d')
# axs[2].set_title('Coordonnées Z')
# axs[2].set_xlabel('Index')
# axs[2].set_ylabel('Z')
# axs[2].legend()
#
# # Afficher les graphiques
# plt.tight_layout()
# plt.show()
