import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import pymap3d as pm
from scipy.io import savemat
import pyproj

def ecef_to_utm19R(x, y, z):
    lats, lons, heights = pm.ecef2geodetic(x, y, z)
    utm19R = pyproj.Proj(proj="utm", zone=19, datum="WGS84")
    eastings, northings = utm19R(lons, lats)
    return eastings, northings, heights

def utm19R_to_ecef(eastings, northings, heights):
    """Convert UTM Zone 19R to ECEF."""
    utm19R = pyproj.Proj(proj="utm", zone=19, datum="WGS84")
    lons, lats = utm19R(eastings, northings, inverse=True)
    x, y, z = pm.geodetic2ecef(lats, lons, heights)
    return x, y, z


def load_and_process_data(path):
    data = sio.loadmat(path)
    days = data['days'].flatten() - 59015
    times = data['times'].flatten()
    datetimes = ((days * 24 * 3600) + times) / 3600

    # Applying the condition
    mask = (datetimes > 25) & (datetimes < 37)

    datetimes = datetimes[mask]
    x, y, z = data['x'].flatten()[mask], data['y'].flatten()[mask], data['z'].flatten()[mask]
    lon, lat, elev = data['lon'].flatten()[mask], data['lat'].flatten()[mask], data['elev'].flatten()[mask]
    return datetimes, x, y, z

def matrix_orthonormalization(C):
    c01 = C[0,:].dot(C[1,:]) / 2.
    c02 = C[0,:].dot(C[2,:]) / 2.
    c12 = C[1,:].dot(C[2,:]) / 2.
    C = np.stack((C[0,:] - c01 * C[1,:] - c02 * C[2,:],
                  C[1,:] - c01 * C[0,:] - c12 * C[2,:],
                  C[2,:] - c02 * C[0,:] - c12 * C[1,:]), 0)
    Cn = np.sqrt((C ** 2).sum(-1))
    C /= Cn[:, None]
    return C

def matrix_to_Tait_Bryan(C):
    C_ = matrix_orthonormalization(C)
    att = np.stack((np.arctan2(C_[...,1, 0], C_[..., 0, 0]),
                    np.arctan2(-C_[...,2, 0], np.sqrt(C_[..., 2, 1]**2 + C_[..., 2, 2]**2)),
                    np.arctan2(C_[..., 2, 1], C_[..., 2, 2])), -1)
    return att

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
for datetimes, x, y, z  in all_data:
    mask = np.isin(datetimes, common_datetimes)
    filtered_data.append((datetimes[mask], x[mask], y[mask], z[mask]))

filtered_data_utm = np.zeros_like(filtered_data)
# Convert the ECEF coordinates to UTM 19R for easier plotting and processing
for idx, data in enumerate(filtered_data):
    datetimes, x, y, z = data
    eastings, northings, heights = ecef_to_utm19R(x, y, z)
    filtered_data_utm[idx] = (datetimes, eastings, northings, heights)

# Reference to Antenna 3
antenna_1_data = filtered_data_utm[0]
antenna_2_data = filtered_data_utm[1]
antenna_3_data = filtered_data_utm[2]
antenna_4_data = filtered_data_utm[3]
relative_positions_body_frame = []

leviers = [
    np.array([10.20, 7.11, 0]),
    np.array([10.24, 2.18, 0]),
    np.array([0, 0, 0]),
    np.array([0, 7.11, 0])
]

# Calcul des positions dans le repère body à partir de l'antenne 3 et des bras de levier
for t in range(len(common_datetimes)):
    antenna_3_position_t = np.array([antenna_3_data[1][t], antenna_3_data[2][t], antenna_3_data[3][t]])
    relative_positions_body_frame.append([antenna_3_position_t + levier for levier in leviers])

# Calculate rotation matrices
rotation_matrices = []
for t in range(len(common_datetimes)):
    A = np.array(relative_positions_body_frame[t])
    B = np.array([(x[t], y[t], z[t]) for _, x, y, z in filtered_data_utm])
    R, _, _ = kabsch_umeyama(A, B)
    rotation_matrices.append(R)

leviers = [
    np.array([-16.6, 2.46, -15.24]),  # transducer1
    np.array([-16.6, 7.39, -15.24]),  # transducer2
    np.array([-6.40, 9.57, -15.24]),  # transducer3
    np.array([-6.40, 2.46, -15.24])   # transducer4
]
# Assuming you have antenna data like this
antennas_data = [antenna_1_data, antenna_2_data, antenna_3_data, antenna_4_data]

positions_transducer = []
std_deviations = np.zeros((len(common_datetimes), 3))  # Initialize array for standard deviations

for t in range(len(common_datetimes)):
    sum_positions = np.zeros(3)  # Initialize with zeros for summing up positions
    temp_positions_t = []  # Temporary storage for positions at time t

    for i in range(4):  # for each antenna
        position_antenna_t = np.array([antennas_data[i][1][t], antennas_data[i][2][t], antennas_data[i][3][t]])
        R = rotation_matrices[t]
        transformed_position = position_antenna_t + R @ leviers[i]
        sum_positions += transformed_position
        temp_positions_t.append(transformed_position)

    avg_position_transducer_t = sum_positions / 4.0  # Taking average
    positions_transducer.append(avg_position_transducer_t)

    # Convert to numpy array for easier manipulation
    temp_positions_t = np.array(temp_positions_t)

    # Compute standard deviation for x, y, and z separately
    std_deviations[t, 0] = np.std(temp_positions_t[:, 0])
    std_deviations[t, 1] = np.std(temp_positions_t[:, 1])
    std_deviations[t, 2] = np.std(temp_positions_t[:, 2])


positions_transducer = np.array(positions_transducer)

# X Position and its standard deviation
plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.plot(common_datetimes, positions_transducer[:, 0], label='Average X Position', color='b')
plt.fill_between(common_datetimes,
                 positions_transducer[:, 0] - std_deviations[:, 0],
                 positions_transducer[:, 0] + std_deviations[:, 0],
                 color='b', alpha=0.2, label='1 Std Dev')
plt.legend()
plt.title('X Position over Time')
plt.ylabel('X Position')

# Y Position and its standard deviation
plt.subplot(3, 1, 2)
plt.plot(common_datetimes, positions_transducer[:, 1], label='Average Y Position', color='g')
plt.fill_between(common_datetimes,
                 positions_transducer[:, 1] - std_deviations[:, 1],
                 positions_transducer[:, 1] + std_deviations[:, 1],
                 color='g', alpha=0.2, label='1 Std Dev')
plt.legend()
plt.title('Y Position over Time')
plt.ylabel('Y Position')

# Z Position and its standard deviation
plt.subplot(3, 1, 3)
plt.plot(common_datetimes, positions_transducer[:, 2], label='Average Z Position', color='r')
plt.fill_between(common_datetimes,
                 positions_transducer[:, 2] - std_deviations[:, 2],
                 positions_transducer[:, 2] + std_deviations[:, 2],
                 color='r', alpha=0.2, label='1 Std Dev')
plt.legend()
plt.title('Z Position over Time')
plt.ylabel('Z Position')
plt.xlabel('Time')

plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 8))

# X standard deviation over time
plt.subplot(3, 1, 1)
plt.plot(common_datetimes, std_deviations[:, 0], label='X Std Dev', color='b')
plt.legend()
plt.title('Standard Deviation of X Position over Time')
plt.ylabel('X Std Dev')

# Y standard deviation over time
plt.subplot(3, 1, 2)
plt.plot(common_datetimes, std_deviations[:, 1], label='Y Std Dev', color='g')
plt.legend()
plt.title('Standard Deviation of Y Position over Time')
plt.ylabel('Y Std Dev')

# Z standard deviation over time
plt.subplot(3, 1, 3)
plt.plot(common_datetimes, std_deviations[:, 2], label='Z Std Dev', color='r')
plt.legend()
plt.title('Standard Deviation of Z Position over Time')
plt.ylabel('Z Std Dev')
plt.xlabel('Time')

plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 8))

# X standard deviation histogram
plt.subplot(3, 1, 1)
plt.hist(std_deviations[:, 0], bins=30, color='b', alpha=0.7)
plt.title('Histogram of X Position Standard Deviations')
plt.ylabel('Frequency')

# Y standard deviation histogram
plt.subplot(3, 1, 2)
plt.hist(std_deviations[:, 1], bins=30, color='g', alpha=0.7)
plt.title('Histogram of Y Position Standard Deviations')
plt.ylabel('Frequency')

# Z standard deviation histogram
plt.subplot(3, 1, 3)
plt.hist(std_deviations[:, 2], bins=30, color='r', alpha=0.7)
plt.title('Histogram of Z Position Standard Deviations')
plt.xlabel('Standard Deviation')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()



# Extraire les coordonnées est, nord et hauteur pour toutes les positions du transducer
eastings_transducer = [pos[0] for pos in positions_transducer]
northings_transducer = [pos[1] for pos in positions_transducer]
heights_transducer = [pos[2] for pos in positions_transducer]

# Convertir ces coordonnées UTM en ECEF
x_transducer, y_transducer, z_transducer = utm19R_to_ecef(eastings_transducer, northings_transducer, heights_transducer)

# Convert ECEF coordinates of transducer to lat, lon, elev
lat_transducer, lon_transducer, elev_transducer = pm.ecef2geodetic(x_transducer, y_transducer, z_transducer)
# Your data
data_dict_transducer = {
    'times': common_datetimes,
    'x': x_transducer,
    'y': y_transducer,
    'z': z_transducer,
    'lat': lat_transducer,
    'lon': lon_transducer,
    'elev': elev_transducer
}

# Save as .mat file
savemat('kabsch_average.mat', data_dict_transducer)


'''
This part allows to do a all dataset of positions, taking into account different lever-arms dx,dy,dz
# savemat('../trajectory/data_transducer_ecef.mat', data_dict_transducer)
import numpy as np
import os
from scipy.io import savemat

increments = np.arange(-1, 1.5, 0.5)  # De -2m à +2m avec un pas de 20cm
leviers = [(-6.40+dx, 9.57 + dy, -15.24 + dz) for dx in increments for dy in increments for dz in increments]


# Assurez-vous que le sous-répertoire existe, sinon créez-le
output_directory = '../trajectory/levier_variations'
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Pour chaque levier, effectuez les mêmes calculs que précédemment
for levier in leviers:
    positions_transducer = []
    for t in range(len(common_datetimes)):
        position_antenne3_t = np.array([antenna_3_data[1][t], antenna_3_data[2][t], antenna_3_data[3][t]])
        R = rotation_matrices[t]
        position_transducer_t = position_antenne3_t + R @ levier
        positions_transducer.append(position_transducer_t)

    eastings_transducer = [pos[0] for pos in positions_transducer]
    northings_transducer = [pos[1] for pos in positions_transducer]
    heights_transducer = [pos[2] for pos in positions_transducer]

    x_transducer, y_transducer, z_transducer = utm19R_to_ecef(eastings_transducer, northings_transducer, heights_transducer)
    lat_transducer, lon_transducer, elev_transducer = pm.ecef2geodetic(x_transducer, y_transducer, z_transducer)

    # Save the transducer data
    filename = f"levier_{levier[0]:.2f}_{levier[1]:.2f}_{levier[2]:.2f}.mat"
    data_dict_transducer = {
        'times': common_datetimes,
        'x': x_transducer,
        'y': y_transducer,
        'z': z_transducer,
        'lat': lat_transducer,
        'lon': lon_transducer,
        'elev': elev_transducer
    }
    savemat(os.path.join(output_directory, filename), data_dict_transducer)
'''

# Convert rotation matrices to Tait-Bryan angles
tait_bryan_angles = [matrix_to_Tait_Bryan(R) for R in rotation_matrices]
yaw = [angles[0] for angles in tait_bryan_angles]
pitch = [angles[1] for angles in tait_bryan_angles]
roll = [angles[2] for angles in tait_bryan_angles]

# Convert the angles to degrees
roll_deg = np.rad2deg(roll)
pitch_deg = np.rad2deg(pitch)
yaw_deg = np.rad2deg(yaw)

# Organize data into a dictionary format
data_to_save = {
    'common_datetimes': common_datetimes,
    'yaw_deg': yaw_deg,
    'pitch_deg': pitch_deg,
    'roll_deg': roll_deg
}

# Save data to .mat file
savemat('attitude_data.mat', data_to_save)

# Plot the Tait-Bryan angles
plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.title('Attitude angles during Bermuda campaign')
plt.scatter(common_datetimes, roll_deg, c='r', s = 2, label='Roll')
plt.ylabel('Roll (degrees)')
plt.legend()

plt.subplot(3, 1, 2)
plt.scatter(common_datetimes, pitch_deg, c='g', s = 2, label='Pitch')
plt.ylabel('Pitch (degrees)')
plt.legend()

plt.subplot(3, 1, 3)
plt.scatter(common_datetimes, yaw_deg, c='b', s = 2, label='Yaw')
plt.xlabel('Time')
plt.ylabel('Yaw (degrees)')
plt.legend()
plt.tight_layout()
plt.show()

# Charger les données des antennes
antenna_colors = ['red', 'green', 'blue', 'purple']
antenna_labels = ['Antenna 1', 'Antenna 2', 'Antenna 3', 'Antenna 4']

# Création du plot
plt.figure(figsize=(12, 8))

# Tracer le transducer avec l'élévation en couleur
sc = plt.scatter(lon_transducer, lat_transducer, c=elev_transducer, cmap='viridis', s=60, label='Transducer')
plt.colorbar(sc, label='Elevation (m)')

# Tracer les antennes avec des couleurs distinctes
for idx, (_, x, y, z) in enumerate(filtered_data):
    latitudes_antennas = [pm.ecef2geodetic(x[t], y[t], z[t])[0] for t in range(len(common_datetimes))]
    longitudes_antennas = [pm.ecef2geodetic(x[t], y[t], z[t])[1] for t in range(len(common_datetimes))]
    plt.scatter(longitudes_antennas, latitudes_antennas, c=antenna_colors[idx], s=30, label=antenna_labels[idx], alpha=0.6)

# Ajouter les légendes et titres
plt.title("Transducer and Antennas Position")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Création des plots
fig = plt.figure(figsize=(18, 8))

# Axes pour les trois subplots à gauche
ax1 = fig.add_subplot(3, 2, 1)
ax2 = fig.add_subplot(3, 2, 3)
ax3 = fig.add_subplot(3, 2, 5)

# Axe pour le plot à droite
ax4 = fig.add_subplot(1, 2, 2)

ax1.plot(common_datetimes, lat_transducer, color='orange', label='Transducer', alpha=0.6)
ax2.plot(common_datetimes, lon_transducer, color='orange', alpha=0.6)
ax3.plot(common_datetimes, elev_transducer, color='orange', alpha=0.6)

# Trajectoire 2D du transducteur à droite
ax4.scatter(lon_transducer, lat_transducer, c='orange', s=30, label='Transducer', alpha=0.8)

# Gauche : Tracer les composantes individuellement pour chaque antenne
for idx, (_, x, y, z) in enumerate(filtered_data):
    latitudes = [pm.ecef2geodetic(x[t], y[t], z[t])[0] for t in range(len(common_datetimes))]
    longitudes = [pm.ecef2geodetic(x[t], y[t], z[t])[1] for t in range(len(common_datetimes))]
    elevations = [pm.ecef2geodetic(x[t], y[t], z[t])[2] for t in range(len(common_datetimes))]

    color = antenna_colors[idx]
    label = antenna_labels[idx]

    ax1.plot(common_datetimes, latitudes, color=color, label=label, alpha=0.6)
    ax2.plot(common_datetimes, longitudes, color=color, alpha=0.6)
    ax3.plot(common_datetimes, elevations, color=color, alpha=0.6)

    # Trajectoire 2D pour chaque antenne à droite
    ax4.scatter(longitudes, latitudes, c=color, s=30, label=label, alpha=0.6)


# Mise en forme des titres, légendes, etc.
ax1.set_title('Latitude')
ax2.set_title('Longitude')
ax3.set_title('Elevation')
ax3.set_xlabel('Time')
ax4.set_title("Transducer and Antennas Position")
ax4.set_xlabel("Longitude")
ax4.set_ylabel("Latitude")

# Mettre en forme et montrer le graphique
ax3.legend()  # Placer la légende sur le subplot d'élévation
ax4.legend(loc="upper right")
plt.tight_layout()
plt.show()
