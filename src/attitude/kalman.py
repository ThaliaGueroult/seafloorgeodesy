import numpy as np
import scipy.io as sio
from scipy.linalg import block_diag
import matplotlib.pyplot as plt

# Chargement des données
data_unit1 = sio.loadmat('../../data/SwiftNav_Data/GPStime_synchro/Unit1-camp_bis_GPStime_synchro.mat')
data_unit2 = sio.loadmat('../../data/SwiftNav_Data/GPStime_synchro/Unit2-camp_bis_GPStime_synchro.mat')
data_unit3 = sio.loadmat('../../data/SwiftNav_Data/GPStime_synchro/Unit3-camp_bis_GPStime_synchro.mat')
data_unit4 = sio.loadmat('../../data/SwiftNav_Data/GPStime_synchro/Unit4-camp_bis_GPStime_synchro.mat')

# Extraction des coordonnées
coords_unit1 = np.array([data_unit1['x'].flatten(), data_unit1['y'].flatten(), data_unit1['z'].flatten()]).T
coords_unit2 = np.array([data_unit2['x'].flatten(), data_unit2['y'].flatten(), data_unit2['z'].flatten()]).T
coords_unit3 = np.array([data_unit3['x'].flatten(), data_unit3['y'].flatten(), data_unit3['z'].flatten()]).T
coords_unit4 = np.array([data_unit4['x'].flatten(), data_unit4['y'].flatten(), data_unit4['z'].flatten()]).T

# Suppression des données invalides
invalid_indices_unit1 = np.isnan(coords_unit1).any(axis=1)
invalid_indices_unit2 = np.isnan(coords_unit2).any(axis=1)
invalid_indices_unit3 = np.isnan(coords_unit3).any(axis=1)
invalid_indices_unit4 = np.isnan(coords_unit4).any(axis=1)
combined_invalid_indices = invalid_indices_unit1 | invalid_indices_unit2 | invalid_indices_unit3 | invalid_indices_unit4
valid_indices = ~combined_invalid_indices

coords_unit1 = coords_unit1[valid_indices]
coords_unit2 = coords_unit2[valid_indices]
coords_unit3 = coords_unit3[valid_indices]
coords_unit4 = coords_unit4[valid_indices]

# Extraction des dates
days = data_unit1['days'].flatten() - 59015
times = data_unit1['times'].flatten()
datetimes = ((days * 24 * 3600) + times)/3600
datetime = datetimes[valid_indices]

# Bras de levier pour chaque antenne
l1 = np.array([16.15, 0, 15.24])
l2 = np.array([16.15, 5, 15.24])
l3 = np.array([6.40, 5, 15.24])
l4 = np.array([6.40, 0, 15.24])

# Initialisation des matrices pour le filtre de Kalman
x_estimated = np.zeros((3, len(datetime)))
x_estimated[:, 0] = coords_unit1[0]

P = np.diag([10, 10, 10])
Q = np.diag([10**2,10**2, 10**2])
R = block_diag(0.02**2, 0.02**2, 0.02**2, 0.02**2, 0.02**2, 0.02**2, 0.02**2, 0.02**2, 0.02**2, 0.02**2, 0.02**2, 0.02**2)
H = np.tile(np.eye(3), (4, 1))

P_list = []
for k in range(1, len(datetime)):
    dt = datetime[k] - datetime[k-1]

    # Étape de prédiction
    x_predicted = x_estimated[:, k-1]
    P_predicted = P + Q

    # Étape de mise à jour
    z = np.hstack([coords_unit1[k], coords_unit2[k], coords_unit3[k], coords_unit4[k]])
    h_x = np.hstack([(x_predicted + l1), (x_predicted + l2), (x_predicted + l3), (x_predicted + l4)])

    y = z - h_x
    S = np.dot(H, np.dot(P_predicted, H.T)) + R
    K = np.dot(P_predicted, np.dot(H.T, np.linalg.inv(S)))
    x_estimated[:, k] = x_predicted + np.dot(K, y)
    P = P_predicted - np.dot(K, np.dot(S, K.T))
    P_list.append(P.copy())

P_array = np.array(P_list)
print(P_array)

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(x_estimated[0, :], x_estimated[1, :], x_estimated[2, :])
ax.plot(data_unit1['x'].flatten(), data_unit1['y'].flatten(),data_unit1['z'].flatten())
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title("Estimation de la trajectoire en 3D")
plt.show()


# Extraction des éléments diagonaux de chaque matrice P stockée
P_00 = [P[0, 0] for P in P_list]
P_11 = [P[1, 1] for P in P_list]
P_22 = [P[2, 2] for P in P_list]

# Tracer ces éléments en fonction du temps
plt.figure(figsize=(10, 6))
plt.scatter(datetime[1:], P_00, label="P[0,0]", s = 1)
plt.scatter(datetime[1:], P_11, label="P[1,1]", s = 1)
plt.scatter(datetime[1:], P_22, label="P[2,2]", s = 1)
plt.xlabel("Temps (Heures depuis le début)")
plt.ylabel("Valeur de la Covariance")
plt.title("Évolution des éléments diagonaux de la matrice P")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.patches import Ellipse
# import matplotlib.animation as animation
#
# def plot_covariance_ellipse(x, P, plt_axes=None):
#     """ Tracer une ellipse de covariance 2D à partir d'une position x et d'une matrice de covariance P."""
#     if plt_axes is None:
#         _, plt_axes = plt.subplots()
#
#     # Extraire la covariance 2D
#     P = P[0:2, 0:2]
#     x = x[0:2]
#
#     # Calculer les valeurs propres et les vecteurs propres
#     vals, vecs = np.linalg.eigh(P)
#     order = vals.argsort()[::-1]
#     vals = vals[order]
#     vecs = vecs[:, order]
#
#     # Calculer l'angle de rotation de l'ellipse
#     theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
#
#     # Tracer l'ellipse
#     width, height = 2 * np.sqrt(vals)
#     ellipse = Ellipse(xy=x, width=width, height=height, angle=theta, edgecolor='r', facecolor='none')
#     plt_axes.add_patch(ellipse)
#     plt_axes.plot(x[0], x[1], 'ro')
#
# # Initialisation de la figure et des axes
# fig, ax = plt.subplots()
# ax.set_xlim([-10, 10])  # Pour exemple, ajuster selon vos besoins
# ax.set_ylim([-10, 10])  # Pour exemple, ajuster selon vos besoins
#
# def update(num, x_estimated, P_array, ax):
#     ax.clear()
#     plot_covariance_ellipse(x_estimated[:, num], P_array[num], plt_axes=ax)
#     ax.set_title(f"Étape: {num}")
#
# ani = animation.FuncAnimation(fig, update, frames=len(x_estimated[0]), fargs=(x_estimated, P_array, ax), interval = 10e-30)
# plt.show()
