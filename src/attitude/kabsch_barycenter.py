import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import math
import pymap3d as pm
from scipy.io import savemat

# Charger et traiter les données
def load_and_process_data(path):
    data = sio.loadmat(path)
    days = data['days'].flatten() - 59015
    times = data['times'].flatten()
    datetimes = ((days * 24 * 3600) + times) / 3600
    x, y, z = data['x'].flatten(), data['y'].flatten(), data['z'].flatten()
    return datetimes, x, y, z

paths = [
    '../../data/SwiftNav_Data/Unit1-camp_bis.mat',
    '../../data/SwiftNav_Data/Unit2-camp_bis.mat',
    '../../data/SwiftNav_Data/Unit3-camp_bis.mat',
    '../../data/SwiftNav_Data/Unit4-camp_bis.mat'
]

all_data = [load_and_process_data(path) for path in paths]
common_datetimes = set(all_data[0][0])
for data in all_data[1:]:
    common_datetimes.intersection_update(data[0])
common_datetimes = sorted(common_datetimes)

filtered_data = []
for datetimes, x, y, z in all_data:
    mask = np.isin(datetimes, common_datetimes)
    filtered_data.append((datetimes[mask], x[mask], y[mask], z[mask]))

levier_A1 = np.array([5.1, 3.022050621710503, 0])
levier_A2 = np.array([5.1, -1.9561518651315093, 0])
levier_A3 = np.array([-5.1, -4.087949378289498, 0])
levier_A4 = np.array([-5.1, 3.022050621710503, 0])

# Calcul du barycentre
def compute_barycenter(data):
    x_total, y_total, z_total = 0, 0, 0
    n = len(data)
    for x, y, z in data:  # Change here
        x_total += x
        y_total += y
        z_total += z
    x_barycenter = x_total / n
    y_barycenter = y_total / n
    z_barycenter = z_total / n
    return x_barycenter, y_barycenter, z_barycenter

barycenter = [compute_barycenter([(x[t], y[t], z[t]) for _, x, y, z in filtered_data]) for t in range(len(common_datetimes))]

relative_positions = []
for t in range(len(common_datetimes)):
    barycenter_t = barycenter[t]
    positions_t = []
    for idx, (_, x, y, z) in enumerate(filtered_data):
        if idx == 0:
            levier = levier_A1
        elif idx == 1:
            levier = levier_A2
        elif idx == 2:
            levier = levier_A3
        elif idx == 3:
            levier = levier_A4
        position_relative_to_barycenter = (levier[0] + barycenter_t[0],
                                           levier[1] + barycenter_t[1],
                                           levier[2] + barycenter_t[2])
        positions_t.append(position_relative_to_barycenter)
    relative_positions.append(positions_t)

colors = ['red', 'green', 'blue', 'purple']  # Couleurs pour les 4 antennes

plt.figure(figsize=(10, 6))

# Scatter des positions pour chaque antenne
for idx in range(4):
    x_values = [positions[idx][0] for positions in relative_positions]
    y_values = [positions[idx][1] for positions in relative_positions]
    plt.scatter(x_values, y_values, color=colors[idx], label=f'Antenne {idx+1}')

plt.title('Positions relatives des antennes par rapport au barycentre')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.show()


def matrix_orthonormalization(C):
    # Produits scalaires
    c01 = C[0,:].dot(C[1,:]) / 2.
    c02 = C[0,:].dot(C[2,:]) / 2.
    c12 = C[1,:].dot(C[2,:]) / 2.

    # Orthogonalisation
    C = np.stack((C[0,:] - c01 * C[1,:] - c02 * C[2,:],
                  C[1,:] - c01 * C[0,:] - c12 * C[2,:],
                  C[2,:] - c02 * C[0,:] - c12 * C[1,:]), 0)
    # Normalisation
    Cn = np.sqrt((C ** 2).sum(-1))
    C /= Cn[:, None]

    return C

def matrix_to_Tait_Bryan(C):
    """ Décomposition non optimale
    ce qui fait qu'il est bon de réorthonormaliser la matrice
    approximativement
    """
    C_ = matrix_orthonormalization(C)
    att = np.stack\
        ((np.arctan2(C_[...,1, 0], C_[..., 0, 0]),
          np.arctan2(-C_[...,2, 0],
                     np.sqrt(C_[..., 2, 1]**2 + C_[..., 2, 2]**2)),
          np.arctan2(C_[..., 2, 1], C_[..., 2, 2])), -1)
    return att

def matrix_from_Tait_Bryan(att):
    cp = np.cos(att[0])
    sp = np.sin(att[0])
    ct = np.cos(att[1])
    st = np.sin(att[1])
    cf = np.cos(att[2])
    sf = np.sin(att[2])

    C = np.stack\
        ((cp * ct, cp * st * sf - sp * cf, cp * st * cf + sp * sf,
          sp * ct, sp * st * sf + cp * cf, sp * st * cf - cp * sf,
          -st, ct * sf, ct * cf), -1)
    C.shape = C.shape[:-1] + (3, 3)
    return C

def kabsch_umeyama(A, B):
    assert A.shape == B.shape
    n, m = A.shape

    EA = np.mean(A, axis=0)
    EB = np.mean(B, axis=0)
    VarA = np.mean(np.linalg.norm(A - EA, axis=1) ** 2)

    H = ((A - EA).T @ (B - EB)) / n
    U, D, VT = np.linalg.svd(H)
    d = np.sign(np.linalg.det(U) * np.linalg.det(VT))
    S = np.diag([1] * (m - 1) + [d])

    R = U @ S @ VT
    c = VarA / np.trace(np.diag(D) @ S)
    t = EA - c * R @ EB

    return R, c, t

# Calculer les matrices de rotation
rotation_matrices = []
for t, antenne_positions in enumerate(relative_positions):
    A = np.array(antenne_positions)
    B = np.array([(x[t], y[t], z[t]) for _, x, y, z in filtered_data])
    R, _, _ = kabsch_umeyama(A, B)
    rotation_matrices.append(R)

# all_relative_positions = np.array([pos for positions in relative_positions for pos in positions])
# print(all_relative_positions.shape)
# # Assuming filtered_data is a list of tuples where each tuple has four elements: a timestamp and x, y, z positions
# all_filtered_positions = np.array([(x, y, z) for _, x, y, z in filtered_data])
# print(all_filtered_positions.shape)
# R_total, _, _ = kabsch_umeyama(all_relative_positions, all_filtered_positions)
#
# att_tot = matrix_to_Tait_Bryan(R_total)
# print(att_tot)

tait_bryan_angles = [matrix_to_Tait_Bryan(R) for R in rotation_matrices]
yaw = [angles[0] for angles in tait_bryan_angles]
pitch = [angles[1] for angles in tait_bryan_angles]
roll = [angles[2] for angles in tait_bryan_angles]

# rotation_matrices = []
# for t in range(len(yaw)):
#     R=matrix_from_Tait_Bryan(np.array([yaw, pitch-np.mean(pitch), roll-np.mean(roll)]))
#     rotation_matrices.append(R)

# Conversion des angles en degrés
roll_deg = np.rad2deg(roll)
pitch_deg = np.rad2deg(pitch)
yaw_deg = np.rad2deg(yaw)

# Créer 3 subplots
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))

# Tracer le Roll
ax1.scatter(common_datetimes, roll_deg, label='Roll', color='r')
ax1.set_title('Roll')
ax1.set_ylabel('Angle (°)')
ax1.set_xlim(25,37.9)
ax1.legend()
ax1.grid(True)

# Tracer le Pitch
ax2.scatter(common_datetimes, pitch_deg, label='Pitch', color='g')
ax2.set_title('Pitch')
ax2.set_ylabel('Angle (°)')
ax2.set_xlim(25,37.9)
ax2.legend()
ax2.grid(True)

# Tracer le Yaw
ax3.scatter(common_datetimes, yaw_deg, label='Yaw', color='b')
ax3.set_title('Yaw')
ax3.set_xlabel('Time')
ax3.set_ylabel('Angle (°)')
ax3.set_xlim(25,37.9)
ax3.legend()
ax3.grid(True)

# Afficher les tracés
plt.tight_layout()
plt.show()


levier_transducteur = np.array([11.5, 5.48, -15.24])

positions_transducteur = []

for t, R in enumerate(rotation_matrices):
    position_rotated = R @ levier_transducteur
    position_global = position_rotated + np.array(barycenter[t])
    positions_transducteur.append(position_global)

# Extraire les positions X, Y et Z pour le graphique ou l'analyse
x_transducteur = [pos[0] for pos in positions_transducteur]
y_transducteur = [pos[1] for pos in positions_transducteur]
z_transducteur = [pos[2] for pos in positions_transducteur]

plt.figure(figsize=(12, 8))

# Plot transducer position
sc_trans = plt.scatter(x_transducteur, y_transducteur, c=z_transducteur, s=50, cmap='viridis', label="Transducteur", alpha=0.8)

# Plot the positions of the antennas
colors = ['red', 'green', 'blue', 'purple']
labels = ['Antenne 1', 'Antenne 2', 'Antenne 3', 'Antenne 4']
for idx in range(4):
    x_values = [positions[idx][0] for positions in relative_positions]
    y_values = [positions[idx][1] for positions in relative_positions]
    plt.scatter(x_values, y_values, color=colors[idx], label=labels[idx], s=20, alpha=0.5)

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Positions du transducteur et des antennes en 2D')
plt.legend()
cbar = plt.colorbar(sc_trans, orientation="vertical", label="Z position")
plt.grid(True)
plt.show()

# Convertir les coordonnées ECEF en coordonnées géodésiques
lats_transducteur, lons_transducteur, elevs_transducteur = pm.ecef2geodetic(x_transducteur, y_transducteur, z_transducteur)

# Création d'un dictionnaire avec les données à enregistrer
data_dict = {
    'times': common_datetimes,
    'x': x_transducteur,
    'y': y_transducteur,
    'z': z_transducteur,
    'lat': lats_transducteur,
    'lon': lons_transducteur,
    'elev': elevs_transducteur
}

# Sauvegarde des données dans un fichier .mat
savemat('../trajectory/data_transduceur.mat', data_dict)

lat_transducteur = []
lon_transducteur = []
elev_transducteur = []

# Convertir les coordonnées ECEF en coordonnées géodésiques
for pos in positions_transducteur:
    lat, lon, elev = pm.ecef2geodetic(pos[0], pos[1], pos[2])
    lat_transducteur.append(lat)
    lon_transducteur.append(lon)
    elev_transducteur.append(elev)

plt.figure(figsize=(12, 8))

# Plot transducer position
sc_trans = plt.scatter(lon_transducteur, lat_transducteur, c=elev_transducteur, s=50, cmap='viridis', label="Transducteur", alpha=0.8)

# Plot the positions of the antennas
colors = ['red', 'green', 'blue', 'purple']
labels = ['Antenne 1', 'Antenne 2', 'Antenne 3', 'Antenne 4']
for idx in range(4):
    lat_values = []
    lon_values = []
    for positions in relative_positions:
        lat, lon, elev = pm.ecef2geodetic(positions[idx][0], positions[idx][1], positions[idx][2])
        lat_values.append(lat)
        lon_values.append(lon)
    plt.scatter(lon_values, lat_values, color=colors[idx], label=labels[idx], s=20, alpha=0.5)

plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Positions du transducteur et des antennes en coordonnées géodésiques')
plt.legend()
cbar = plt.colorbar(sc_trans, orientation="vertical", label="Élévation (m)")
plt.grid(True)
plt.show()


# # Extraction des positions
# positions_body = np.array(relative_positions).reshape(-1, 3)
# print(positions_body)
# positions_nav = np.array([[(x[t], y[t], z[t]) for _, x, y, z in filtered_data] for t in range(len(common_datetimes))]).reshape(-1, 3)
# plt.figure(figsize=(14, 7))
#
# # Positions dans le repère body
# ax1 = plt.subplot(1, 2, 1)
# sc1 = ax1.scatter(positions_body[:, 0], positions_body[:, 1], c=positions_body[:, 2], cmap='viridis', s=15)
# ax1.set_title("Positions dans le repère Body")
# ax1.set_xlabel('X')
# ax1.set_ylabel('Y')
# cbar1 = plt.colorbar(sc1, ax=ax1, orientation='vertical')
# cbar1.set_label('Z')
#
# # Positions dans le repère navigation
# ax2 = plt.subplot(1, 2, 2)
# sc2 = ax2.scatter(positions_nav[:, 0], positions_nav[:, 1], c=positions_nav[:, 2], cmap='viridis', s=15)
# ax2.set_title("Positions dans le repère Navigation")
# ax2.set_xlabel('X')
# ax2.set_ylabel('Y')
# cbar2 = plt.colorbar(sc2, ax=ax2, orientation='vertical')
# cbar2.set_label('Z')
#
# plt.tight_layout()
# plt.show()
