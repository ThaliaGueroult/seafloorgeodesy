import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import pymap3d as pm
from scipy.io import savemat
import pyproj

def ecef_to_utm19R(x, y, z):
    # Convert ECEF to Lat, Lon, Height first
    lats, lons, heights = pm.ecef2geodetic(x, y, z)

    # Define the UTM19R projection
    utm19R = pyproj.Proj(proj="utm", zone=19, datum="WGS84")
    eastings, northings = utm19R(lons, lats)

    return eastings, northings, heights

def utm19R_to_ecef(eastings, northings, heights):
    """Convert UTM Zone 19R to ECEF."""
    utm19R = pyproj.Proj(proj="utm", zone=19, datum="WGS84")
    lons, lats = utm19R(eastings, northings, inverse=True)
    x, y, z = pm.geodetic2ecef(lats, lons, heights)
    return x, y, z

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

# Convert the ECEF coordinates to UTM 19R
for idx, data in enumerate(filtered_data):
    datetimes, x, y, z = data
    eastings, northings, heights = ecef_to_utm19R(x, y, z)
    filtered_data[idx] = (datetimes, eastings, northings, heights)


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
    R, _, t = kabsch_umeyama(A, B)
    rotation_matrices.append(R)

tait_bryan_angles = [matrix_to_Tait_Bryan(R) for R in rotation_matrices]
yaw = [angles[0] for angles in tait_bryan_angles]
pitch = [angles[1] for angles in tait_bryan_angles]
roll = [angles[2] for angles in tait_bryan_angles]

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


# For the transducer position, when using the rotation matrices:
positions_transducteur = []
for t, R in enumerate(rotation_matrices):
    position_rotated = R @ levier_transducteur
    position_global = position_rotated + np.array(barycenter[t])
    positions_transducteur.append(position_global)


easting_transducteur = [pos[0] for pos in positions_transducteur]
northing_transducteur = [pos[1] for pos in positions_transducteur]
height_transducteur = [pos[2] for pos in positions_transducteur]

plt.figure(figsize=(12, 8))

# Plot transducer position
sc_trans = plt.scatter(easting_transducteur, northing_transducteur, c=height_transducteur, s=50, cmap='viridis', label="Transducteur", alpha=0.8)
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
# Convert back UTM 19R to ECEF
def utm19R_to_ecef(eastings, northings, heights):
    """Convert UTM Zone 19R to ECEF."""
    utm19R = pyproj.Proj(proj="utm", zone=19, datum="WGS84")
    lons, lats = utm19R(eastings, northings, inverse=True)
    x, y, z = pm.geodetic2ecef(lats, lons, heights)
    return x, y, z

x_ecef, y_ecef, z_ecef = utm19R_to_ecef(
    np.array([pos[0] for pos in positions_transducteur]),
    np.array([pos[1] for pos in positions_transducteur]),
    np.array([pos[2] for pos in positions_transducteur])
)

# Convert ECEF coordinates to lat, lon, elev
lat, lon, elev = pm.ecef2geodetic(x_ecef, y_ecef, z_ecef)

# Save the data
data_dict_ecef = {
    'times': common_datetimes,
    'x': x_ecef,
    'y': y_ecef,
    'z': z_ecef,
    'lat': lat,
    'lon': lon,
    'elev': elev
}

savemat('../trajectory/data_transduceur_ecef.mat', data_dict_ecef)
