#!/usr/bin/python3
import scipy.io as sio
from scipy.optimize import least_squares
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import scipy
import pymap3d as pm

def load_and_process_data(path):
    """Charge et traite les données d'une unité."""
    data = sio.loadmat(path)
    days = data['days'].flatten() - 59015
    times = data['times'].flatten()
    datetimes = ((days * 24 * 3600) + times) / 3600
    x, y, z = data['x'].flatten(), data['y'].flatten(), data['z'].flatten()
    return datetimes, x, y, z

def filter_data_by_common_times(A, common_times):
    mask = np.isin(A[0], common_times)
    return A[0][mask], A[1][mask], A[2][mask], A[3][mask]

def error_function(variables, xa, ya, za, da):
    x, y, z = variables
    distances = np.sqrt((xa - x)**2 + (ya - y)**2 + (za - z)**2)
    return (distances - da)

def get_bounds(A1_filtered, A2_filtered, A3_filtered, A4_filtered, i):
    uncertainty = 1.0  # 1 meter uncertainty

    x_bounds = [
        min([A1_filtered[1][i] + 16.6, A2_filtered[1][i] + 16.6, A3_filtered[1][i] + 6.40, A4_filtered[1][i] + 6.40]) - uncertainty,
        max([A1_filtered[1][i] + 16.6, A2_filtered[1][i] + 16.6, A3_filtered[1][i] + 6.40, A4_filtered[1][i] + 6.40]) + uncertainty
    ]

    y_bounds = [
        min([A1_filtered[2][i] - 2.46, A2_filtered[2][i] - 7.39, A3_filtered[2][i] - 9.57, A4_filtered[2][i] - 2.46]) - uncertainty,
        max([A1_filtered[2][i] - 2.46, A2_filtered[2][i] - 7.39, A3_filtered[2][i] - 9.57, A4_filtered[2][i] - 2.46]) + uncertainty
    ]

    z_bounds = [
        min([A1_filtered[3][i] + 15.24, A2_filtered[3][i] + 15.24, A3_filtered[3][i] + 15.24, A4_filtered[3][i] + 15.24]) - uncertainty,
        max([A1_filtered[3][i] + 15.24, A2_filtered[3][i] + 15.24, A3_filtered[3][i] + 15.24, A4_filtered[3][i] + 15.24]) + uncertainty
    ]

    return ([x_bounds[0], y_bounds[0], z_bounds[0]], [x_bounds[1], y_bounds[1], z_bounds[1]])


paths = [
    '../../data/SwiftNav_Data/Unit1-camp_bis.mat',
    '../../data/SwiftNav_Data/Unit2-camp_bis.mat',
    '../../data/SwiftNav_Data/Unit3-camp_bis.mat',
    '../../data/SwiftNav_Data/Unit4-camp_bis.mat'
]


# Chargez les données pour chaque antenne
A1 = load_and_process_data(paths[0])
A2 = load_and_process_data(paths[1])
A3 = load_and_process_data(paths[2])
A4 = load_and_process_data(paths[3])
common_times = np.intersect1d(A1[0], np.intersect1d(A2[0], np.intersect1d(A3[0], A4[0])))

A1_filtered = filter_data_by_common_times(A1, common_times)
A2_filtered = filter_data_by_common_times(A2, common_times)
A3_filtered = filter_data_by_common_times(A3, common_times)
A4_filtered = filter_data_by_common_times(A4, common_times)

# Liste des datetime que vous souhaitez traiter
datetime_list = []

estimated_positions = []

xa_list, ya_list, za_list, da_list = [], [], [], []
for path in paths:
    datetimes, xa, ya, za = load_and_process_data(path)
    datetime_list.append(datetimes)
    xa_list.append(xa)  # Sélectionner la valeur correspondante au datetime
    ya_list.append(ya)
    za_list.append(za)


# levier = {
#     "antenne 1": np.array([-16.53793082,   2.24612876, -15.17258002]),
#     "antenne 2": np.array( [-16.29752871,   7.09213979, -14.93735477]),
#     "antenne 3": np.array([ -5.60312943,   8.84829343, -14.63103173]),
#     "antenne 4": np.array([ -5.8376399,    2.10820861, -15.22444008])
# }
#
# da1 = np.sqrt(16.53**2 + 2.24**2 + 15.17**2)
# da2 = np.sqrt(16.29**2 + 7.09**2 + 14.93 **2)
# da3 = np.sqrt(5.60**2 + 8.84**2 + 14.63**2)
# da4 = np.sqrt(5.84**2 + 2.10**2 + 15.22**2)

da1 = np.sqrt(16.6**2 + 2.46**2 + 15.24**2)
da2 = np.sqrt(16.6**2 + 7.39**2 + 15.24**2)
da3 = np.sqrt(6.40**2 + 9.57**2 + 15.24**2)
da4 = np.sqrt(6.40**2 + 2.46**2 + 15.24**2)
da = np.array([da1, da2, da3, da4])

# Create a 3D plot for the positions
fig_positions = plt.figure()
ax = fig_positions.add_subplot(111, projection='3d')

estimated_positions = np.zeros((len(common_times), 3))
residuals_list = np.zeros(len(common_times))

for i in tqdm(range(len(common_times)), desc="Processing datetimes"):

    xa = np.array([A1_filtered[1][i], A2_filtered[1][i], A3_filtered[1][i], A4_filtered[1][i]])
    ya = np.array([A1_filtered[2][i], A2_filtered[2][i], A3_filtered[2][i], A4_filtered[2][i]])
    za = np.array([A1_filtered[3][i], A2_filtered[3][i], A3_filtered[3][i], A4_filtered[3][i]])

    bounds = get_bounds(A1_filtered, A2_filtered, A3_filtered, A4_filtered, i)
    x = np.array(bounds[0])
    result = least_squares(error_function, x, args=(xa, ya, za, da))

    # print('\nda',da)
    residuals = error_function(result.x, xa, ya, za, da)

    sum_of_squares = np.sum(np.square(residuals))
    mean_square_error = sum_of_squares / len(residuals)
    rmse = np.sqrt(mean_square_error)
    residuals_list[i] = np.sum(residuals ** 2)

    estimated_positions[i, :] = result.x


# 1. Filtrez les résidus:
threshold = np.percentile(residuals_list, 95)  # par exemple, filtrer les 5% les plus élevés
filtered_residuals = residuals_list[residuals_list < threshold]
fig_residuals_filtered = plt.figure()
ax_residuals_filtered = fig_residuals_filtered.add_subplot(111)
ax_residuals_filtered.hist(filtered_residuals, label="Filtered Residuals", color='b', bins=50)
ax_residuals_filtered.set_ylabel("Residuals")
ax_residuals_filtered.set_title("Filtered Residuals over Time")
ax_residuals_filtered.legend()


# Convert estimated_positions from ECEF to geodetic
lat_est, lon_est, elev_est = pm.ecef2geodetic(estimated_positions[:, 0], estimated_positions[:, 1], estimated_positions[:, 2])

# 2. Take a random moment and plot the next 10 points
random_index = np.random.randint(0, len(common_times) - 10)

fig_random_3D = plt.figure()
ax_random_3D = fig_random_3D.add_subplot(111, projection='3d')
colors = ['b', 'g', 'r', 'm']

for j in range(random_index, random_index + 10):
    ax_random_3D.scatter(lon_est[j], lat_est[j], elev_est[j], c='k', marker='x')
    for idx, A in enumerate([A1_filtered, A2_filtered, A3_filtered, A4_filtered]):
        lat, lon, elev = pm.ecef2geodetic(A[1][j], A[2][j], A[3][j])
        ax_random_3D.scatter(lon, lat, elev, c=colors[idx], marker='o')

# Adding axis labels and title
ax_random_3D.set_xlabel('Longitude')
ax_random_3D.set_ylabel('Latitude')
ax_random_3D.set_zlabel('Elevation')
ax_random_3D.set_title(f"Time: {common_times[random_index]}")

# 3. Plot lat, lon, elev of the estimated position in 2D
fig_2D, ax_2D = plt.subplots()
sc = ax_2D.scatter(lon_est, lat_est, c=elev_est, cmap='viridis')
for idx, A in enumerate([A1_filtered, A2_filtered, A3_filtered, A4_filtered]):
    lat, lon, _ = pm.ecef2geodetic(A[1], A[2], A[3])
    ax_2D.scatter(lon, lat, c=colors[idx], marker='o')
ax_2D.set_xlabel('Longitude')
ax_2D.set_ylabel('Latitude')
cbar = plt.colorbar(sc)
cbar.set_label('Elevation')

plt.show()

# Adaptez vos données à partir de estimated_positions
transducer_positions = estimated_positions
x_transducer = np.column_stack((common_times, transducer_positions))

# Convertir les coordonnées ECEF en coordonnées géodésiques
lat, lon, elev = pm.ecef2geodetic(transducer_positions[:, 0],
                                  transducer_positions[:, 1],
                                  transducer_positions[:, 2],
                                  ell=None, deg=True)
print(elev)
# Préparer le dictionnaire data_to_save
data_to_save = {
    'x': transducer_positions[:, 0],
    'y': transducer_positions[:, 1],
    'z': transducer_positions[:, 2],
    'times': common_times,
    'lat': lat,
    'lon': lon,
    'elev': elev
}
from scipy.io import savemat

# Spécifiez le chemin d'accès et le nom de votre fichier .mat
filename = "transducer_positions.mat"

# Sauvegardez le dictionnaire dans un fichier .mat
savemat(filename, data_to_save)



# A1_filtered = [0,9,11,9]
# A2_filtered = [0,9,9,11]
# A3_filtered = [0,9,9,10]
# A3_filtered = [0,10,10,10]
# da1 = np.sqrt((9-1)**2+(11-1)**2+(9-1)**2)
# print('error function',np.sqrt(error_function((1,1,1),9,11,9,0)))
# da2 = np.sqrt((9-1)**2+(9-1)**2+(11-1)**2)
# da3 = np.sqrt((9-1)**2+(9-1)**2+(10-1)**2)
# da4 = np.sqrt((10-1)**2+(10-1)**2+(10-1)**2)
# da = np.array([da1, da2, da3, da4])
# # for i in tqdm(range(len(common_times)), desc="Processing datetimes"):
# for i in tqdm(range(1), desc="Processing datetimes"):
#     x = np.array([10, 10, 10])
#
#     xa = np.array([9, 9, 9, 10])
#     ya = np.array([11, 9, 9, 10])
#     za = np.array([9, 11, 10,10])
#
#     result = least_squares(error_function, x, args=(xa, ya, za, da))
#     print(da)
#     residuals = error_function(result.x, xa, ya, za, da)
#     print(residuals)
#     print(result)
#     residuals_list[i] = np.sum(residuals ** 2)
#
#     estimated_positions[i, :] = result.x
# print(estimated_positions)

# i = 1507
# x = np.array([A4_filtered[1][i], A4_filtered[2][i], A4_filtered[3][i]])
#
# xa = np.array([A1_filtered[1][i], A2_filtered[1][i], A3_filtered[1][i], A4_filtered[1][i]])
# ya = np.array([A1_filtered[2][i], A2_filtered[2][i], A3_filtered[2][i], A4_filtered[2][i]])
# za = np.array([A1_filtered[3][i], A2_filtered[3][i], A3_filtered[3][i], A4_filtered[3][i]])
# print('/npositions',xa,ya,za)
# result = least_squares(error_function, x, args=(xa, ya, za, da))
# print('\nda',da)
# residuals = error_function(result.x, xa, ya, za, da)
# residuals_list[i] = np.sum(residuals ** 2)
#
# estimated_positions[i, :] = result.x
# # Erreur associée en RMS
# misfit = error_function(result.x, xa, ya, za, da)
# print(misfit)
# rms = np.sqrt(misfit/4)
# print(f"Estimated position [x,y,z] for {int(datetimes[i]*100)/100}s :", result.x)
# print(f"RMSE : ", rms, "m\n")
#
# # Position estimée
# estimated_x, estimated_y, estimated_z = result.x
#
# # Calcul de la distance pour chaque antenne
# distances = np.zeros(4)  # Initialise un tableau pour stocker les distances
# for j in range(4):
#     distances[j] = np.sqrt((xa[j] - estimated_x)**2 + (ya[j] - estimated_y)**2 + (za[j] - estimated_z)**2)
#
# print("Distances entre la position estimée et les antennes :")
# for j in range(4):
#     print(f"Distance à l'antenne {j+1} : {distances[j]} m")
#
# # Calculer l'erreur entre da et la distance estimée
# distance_errors = np.zeros(4)
# for j in range(4):
#     distance_errors[j] = distances[j] - da[j]
#
# print("Erreurs entre la distance théorique et la distance réelle :")
# for j in range(4):
#     print(f"Erreur pour l'antenne {j+1} : {distance_errors[j]} m")
#
# # Calculer l'erreur entre da et la distance estimée et la mettre au carré
# squared_distance_errors = np.zeros(4)
# for j in range(4):
#     squared_distance_errors[j] = (distances[j] - da[j])**2
#
# # Calculer le RMSE
# RMSE = np.sqrt(np.sum(squared_distance_errors) / 4)
#
# print(f"Erreur quadratique moyenne (RMSE) : {RMSE} m")
