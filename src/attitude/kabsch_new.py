import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.io import savemat
import pymap3d as pm

# Paramètres WGS84
f1 = 298.257223563
f = 1. / f1
e2 = (2. - f) * f
a = 6378137.0

def load_and_process_data(path):
    """Charge et traite les données d'une unité."""
    data = sio.loadmat(path)
    days = data['days'].flatten() - 59015
    times = data['times'].flatten()
    datetimes = ((days * 24 * 3600) + times) / 3600
    x, y, z = data['x'].flatten(), data['y'].flatten(), data['z'].flatten()
    lon = np.deg2rad(data['lon'].flatten())
    lat = np.deg2rad(data['lat'].flatten())
    elev = data['elev'].flatten()
    return datetimes, x, y, z, lon, lat, elev

paths = [
    '../../data/SwiftNav_Data/Unit1-camp_bis.mat',
    '../../data/SwiftNav_Data/Unit2-camp_bis.mat',
    '../../data/SwiftNav_Data/Unit3-camp_bis.mat',
    '../../data/SwiftNav_Data/Unit4-camp_bis.mat'
]

all_datetimes, all_xs, all_ys, all_zs, all_lons, all_lats, all_elevs = [], [], [], [], [], [], []

for path in paths:
    datetimes, x, y, z, lon, lat, elev = load_and_process_data(path)
    all_datetimes.append(datetimes)
    all_xs.append(x)
    all_ys.append(y)
    all_zs.append(z)
    all_lons.append(lon)
    all_lats.append(lat)
    all_elevs.append(elev)

print(lon)

def lonlath_to_ecef(lonlath):
    cl, sl = np.cos(lonlath[...,0]), np.sin(lonlath[...,0])
    cp, sp = np.cos(lonlath[...,1]), np.sin(lonlath[...,1])
    N = a / np.sqrt(1. - e2 * (sp**2))
    x_b_e = np.empty(lonlath.shape, np.float64)
    x_b_e[...,0] = (N + lonlath[...,2]) * cp * cl
    x_b_e[...,1] = (N + lonlath[...,2]) * cp * sl
    x_b_e[...,2] = (N * (1 - e2) + lonlath[...,2]) * sp
    return x_b_e

def lonlath_to_ned(lonlath0, lonlath):
    x0 = lonlath_to_ecef(lonlath0)
    x = lonlath_to_ecef(lonlath)
    m = np.empty((3,3), np.float64)
    cp, sp = np.cos(lonlath0[0,1]), np.sin(lonlath0[0,1])
    cl, sl = np.cos(lonlath0[0,0]), np.sin(lonlath0[0,0])
    m[0] = [-sp*cl, -sl, -cp*cl]
    m[1] = [-sp*sl,  cl, -cp*sl]
    m[2] = [ cp,     0., -sp]
    x = np.einsum("ij,...j->...i", m, x - x0)
    return x

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

def lonlat_to_ned_coords(all_lons, all_lats, all_elevs, lonlath0):
    all_ned_coords = []
    for lon, lat, elev in zip(all_lons, all_lats, all_elevs):
        lonlath = np.vstack((lon, lat, elev)).T
        ned_coords = lonlath_to_ned(lonlath0, lonlath)
        all_ned_coords.append(ned_coords)
    return all_ned_coords

lonlath0 = np.array([[all_lons[3][0], all_lats[3][0], all_elevs[3][0]]])
all_ned = lonlat_to_ned_coords(all_lons, all_lats, all_elevs, lonlath0)

bras_de_levier = [
    [0, 0, 0],
    [-0.272, 0.030, -0.041],
    [-0.293, -0.337, -0.079],
    [0.296, 0.296, -0.039]
]

all_ned_body = []

for ned_coords, levier in zip(all_ned, bras_de_levier):
    coords_body = ned_coords + levier  # Ajout des bras de levier
    all_ned_body.append(coords_body)

# Calcul de l'orientation

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

lonlath0 = np.array([[all_lons[3][0], all_lats[3][0], all_elevs[3][0]]])
all_ned = lonlat_to_ned_coords(all_lons, all_lats, all_elevs, lonlath0)

# Coordonnées dans le repère body à l'aide des bras de levier
bras_de_levier = [
    [0, 0, 0],        # Unit 1
    [-0.272, 0.030, -0.041],  # Unit 2
    [-0.293, -0.337, -0.079],  # Unit 3
    [0.296, 0.296, -0.039]   # Unit 4
]

all_ned_body = []
for ned_coords, levier in zip(all_ned, bras_de_levier):
    coords_body = ned_coords - levier
    all_ned_body.append(coords_body)

all_rotations = []

# Assumons que all_ned et all_ned_body ont la même longueur
for t in range(len(all_datetimes[0])):
    # Pour chaque datetime, formons les matrices de source et cible pour Kabsch-Umeyama
    source_matrix = np.array([
        all_ned_body[0][t],
        all_ned_body[1][t],
        all_ned_body[2][t],
        all_ned_body[3][t]
    ])

    target_matrix = np.array([
        all_ned[0][t],
        all_ned[1][t],
        all_ned[2][t],
        all_ned[3][t]
    ])

    # Appliquons la transformation Kabsch-Umeyama pour cette datetime
    R, _, _ = kabsch_umeyama(source_matrix, target_matrix)
    # Stockons la matrice de rotation pour ce point de datetime
    all_rotations.append(R)

# À ce stade, all_rotations est une liste de matrices de rotation pour chaque point de datetime

# Si vous voulez les convertir en angles de Tait-Bryan, vous pouvez le faire comme ceci :

all_angles = [matrix_to_Tait_Bryan(R) for R in all_rotations]

roll_angles = [angles[0] for angles in all_angles]
pitch_angles = [angles[1] for angles in all_angles]
yaw_angles = [angles[2] for angles in all_angles]


all_datetimes = all_datetimes[0]
print(len(all_datetimes))

plt.figure(figsize=(15, 10))

# Roulis
plt.subplot(3, 1, 1)
plt.plot(all_datetimes, np.degrees(roll_angles), label="Roulis", color='r')
plt.ylabel('Angle (radians)')
plt.title('Roulis vs. Temps')
plt.xlim(25,40.9)
plt.legend()
plt.grid(True)

# Tangage
plt.subplot(3, 1, 2)
plt.plot(all_datetimes, np.degrees(pitch_angles), label="Tangage", color='g')
plt.ylabel('Angle (radians)')
plt.title('Tangage vs. Temps')
plt.xlim(25,40.9)
plt.legend()
plt.grid(True)

# Lacet
plt.subplot(3, 1, 3)
plt.plot(all_datetimes, np.degrees(yaw_angles), label="Lacet", color='b')
plt.xlabel('Temps (heure)')
plt.ylabel('Angle (radians)')
plt.title('Lacet vs. Temps')
plt.xlim(25,40.9)
plt.legend()
plt.grid(True)

# Montrer les plots
plt.tight_layout()
plt.show()

# Bras de levier du transducteur par rapport à l'antenne 4 dans le repère Body
levier_transducteur = np.array([-6.40, 2.3, 15.24])

# Positions ECEF de l'antenne 4
positions_antenne4_ecef = np.column_stack((all_xs[3], all_ys[3], all_zs[3]))

# Liste pour stocker les positions ECEF du transducteur
positions_transducteur_ecef = []

# Pour chaque matrice de rotation et position ECEF de l'antenne 4
for R, position_antenne4 in zip(all_rotations, positions_antenne4_ecef):
    # Convertir la position du transducteur dans le repère Body vers ECEF
    position_transducteur_relative_ecef = R @ levier_transducteur

    # Calculer la position ECEF du transducteur
    position_transducteur_ecef = position_transducteur_relative_ecef + position_antenne4

    positions_transducteur_ecef.append(position_transducteur_ecef)

# Convertir en un array NumPy pour une utilisation ultérieure
positions_transducteur_ecef = np.array(positions_transducteur_ecef)
# Extraction des coordonnées x, y, et z du transducteur
xs_transducteur = positions_transducteur_ecef[:, 0]
ys_transducteur = positions_transducteur_ecef[:, 1]
zs_transducteur = positions_transducteur_ecef[:, 2]

print(len(xs_transducteur))

# Création d'une figure et d'axes
fig, ax = plt.subplots(figsize=(10, 8))

# Tracé des positions du transducteur avec la couleur basée sur la coordonnée z
sc = ax.scatter(xs_transducteur, ys_transducteur, c=zs_transducteur, cmap='viridis', s=10, label='Transducteur')

# Tracé des trajectoires des antennes avec une couleur fixe
colors = ['red', 'blue', 'green', 'purple']
for i in range(4):  # Nous supposons qu'il y a 4 antennes
    ax.scatter(all_xs[i], all_ys[i], color=colors[i], s=1, label=f'Antenne {i+1}')

# Configuration des labels, des titres et de la légende
ax.set_title("Trajectoire des antennes et position du transducteur en ECEF")
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.legend(loc="upper right")

# Ajout de la colorbar pour indiquer la valeur de Z du transducteur
cb = plt.colorbar(sc)
cb.set_label('Z (m) du transducteur')

# Affichage du graphique
plt.tight_layout()
plt.show()

# Convertir les coordonnées ECEF en coordonnées géodésiques
lats_transducteur, lons_transducteur, elevs_transducteur = pm.ecef2geodetic(xs_transducteur, ys_transducteur, zs_transducteur)

# Création d'un dictionnaire avec les données à enregistrer
data_dict = {
    'times': all_datetimes,
    'x': xs_transducteur,
    'y': ys_transducteur,
    'z': zs_transducteur,
    'lat': lats_transducteur,
    'lon': lons_transducteur,
    'elev': elevs_transducteur
}

# Sauvegarde des données dans un fichier .mat
savemat('../trajectory/data_transduceur.mat', data_dict)
