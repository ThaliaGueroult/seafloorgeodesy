#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from pymap3d import geodetic2ecef, ecef2geodetic
import os
from multiprocessing import Pool
import time

# Load the Earth-to-Space Volume (ESV) matrix from a .mat file
# Path to the folder containing the .mat file
folder_path = '../esv/esv_table_without_tol/global_table_interp'
matrix_file_path = os.path.join(folder_path, 'global_table_esv.mat')

# Load the matrix data
data = sio.loadmat(matrix_file_path)

# Flatten the arrays for easier manipulation
dz_array = data['distance'].flatten()
angle_array = data['angle'].flatten()
esv_matrix = data['matrice']


def create_grid_around(coord, extension=5.0):
    step = 1.0 # 1 mm
    range_vals = np.arange(-extension, extension + step, step)
    print(range_vals)
    # Créez une grille 3D
    xx, yy, zz = np.meshgrid(range_vals, range_vals, range_vals, indexing='ij')
    x_center,y_center, z_center = geodetic2ecef(coord[0],coord[1],coord[2])
    # Ajustez la grille par rapport à 'coord'
    xx += x_center
    yy += y_center
    zz += z_center

    # Reshape pour avoir une liste de coordonnées
    grid_points = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))

    return grid_points

# Function to convert source and receiver coordinates to declination angle and elevation
def xyz2dec(xyzS, xyzR):
    '''
    Inputs:
        xyzS - Source coordinates [longitude, latitude, altitude]
        xyzR - Receiver coordinates [longitude, latitude, altitude]

    Outputs:
        dz - Vertical distance between source and receiver (m)
        beta - Elevation angle from source to receiver (°)
    '''
    # Calculate the intermediate point coordinates
    xs, ys, zs = geodetic2ecef(xyzS[0], xyzS[1], xyzS[2])
    xr, yr, zr = geodetic2ecef(xyzR[0], xyzR[1], xyzR[2])

    # Calculate the vertical distance
    dz = xyzR[2] - xyzS[2]
    dx = xr - xs
    dy = yr - ys

    dh = np.sqrt(dx**2 + dy**2)

    # Calculate the elevation angle
    beta = np.degrees(np.arctan2(dz, dh))
    beta = np.abs(beta)
    return beta,dz

# Function to find closest ESV value based on dz and beta
def find_esv_grid(dz_array, angle_array, esv_matrix, beta, dz):
    """
    This function finds the closest ESV value given dz and beta.
    Input:
        dz_array: array of dz values from the ESV matrix
        angle_array: array of beta values from the ESV matrix
        esv_matrix: the ESV matrix
        beta: the calculated beta
        dz: the calculated dz
    Output:
        The closest ESV value
    """
    # Utilisez le broadcasting pour trouver les indices des valeurs les plus proches
    idx_closest_dz = np.argmin(np.abs(dz_array[:, None] - dz), axis=0)
    idx_closest_beta = np.argmin(np.abs(angle_array[:, None] - beta), axis=0)

    # Récupérez les valeurs ESV les plus proches pour chaque paire (beta, dz)
    closest_esv = esv_matrix[idx_closest_dz, idx_closest_beta]
    return closest_esv


def calculate_travel_times_grid(trajectory, grid_points):
    num_grid_points = len(grid_points)
    num_trajectory_points = len(trajectory)

    # Convert trajectory and grid_points to numpy arrays for vectorized operations
    trajectory = np.array(trajectory)
    grid_points = np.array(grid_points)

    # Preallocate array for speed
    travel_times = np.zeros((num_trajectory_points, num_grid_points))

    # Convert trajectory to ECEF coordinates
    xs, ys, zs = geodetic2ecef(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2])

    for j, xyzR in enumerate(grid_points):
        xr, yr, zr = xyzR

        # Calculate beta and dz for each trajectory point with respect to the current grid point
        beta, dz = xyz2dec(trajectory, xyzR)  # Update this function to handle arrays

        # Find ESV values for the given beta and dz
        esv = find_esv_grid(dz_array, angle_array, esv_matrix, beta, dz)  # Update this function to handle arrays

        # Calculate the squared distance from each trajectory point to the current grid point
        distances_squared = (xs - xr)**2 + (ys - yr)**2 + (zs - zr)**2

        # Calculate travel time using ESV
        travel_times[:, j] = np.sqrt(distances_squared) / esv

    return travel_times


def GNSS_trajectory(lat, lon, elev):
    trajectory = list(zip(lat, lon, elev))
    return trajectory

def calculate_difference_for_point(grid_points, lat, lon, elev, time_GNSS, time_DOG, acoustic_DOG):
    traj_reel = GNSS_trajectory([lat], [lon], [elev])
    slant_ranges = calculate_travel_times_grid(traj_reel, grid_points)
    interpolated_acoustic_DOG = np.interp(time_GNSS, time_DOG, acoustic_DOG)
    difference_data = slant_ranges - interpolated_acoustic_DOG
    return np.sqrt(np.mean(difference_data**2, axis=0))

def calculate_differences(grid_points, lat, lon, elev, time_GNSS, time_DOG, acoustic_DOG):
    num_grid_points = len(grid_points)

    differences = np.zeros(num_grid_points)

    traj_reel = GNSS_trajectory([lat], [lon], [elev])

    xs, ys, zs = geodetic2ecef(traj_reel[0][0], traj_reel[0][1], traj_reel[0][2])

    interpolated_acoustic_DOG = np.interp(time_GNSS, time_DOG, acoustic_DOG)

    for j, grid_point in enumerate(grid_points):
        beta, dz = xyz2dec(traj_reel[0], grid_point)
        beta = np.array(beta)

        esv = find_esv_grid(dz_array, angle_array, esv_matrix, beta, dz)

        xr, yr, zr = geodetic2ecef(grid_point[0], grid_point[1], grid_point[2])
        travel_time = np.sqrt((xs-xr)**2 + (ys-yr)**2 + (zs-zr)**2) / esv

        # plt.scatter(time_GNSS, travel_time)
        # plt.scatter(time_DOG, acoustic_DOG)
        # plt.show()

        difference_data = travel_time - interpolated_acoustic_DOG
        differences[j] = np.sqrt(np.mean(difference_data ** 2))

    return differences


if __name__ == '__main__':
    # Vos données et fonctions préliminaires
    xyzR =  [  31.46356396,  291.2985875,  5190.77000034] #GNSS1
    # xyzR = [31.46355667,291.29858588, 5189.86379623] #GNSS2
    # xyzR = [31.4635628, 291.29857683, 5189.23243428] #GNSS3
    # xyzR = [31.46357218, 291.29857935, 5190.46271844] #GNSS4


    data_unit = sio.loadmat('../../data/SwiftNav_Data/Unit1-camp_bis.mat')
    days = data_unit['days'].flatten() - 59015
    times = data_unit['times'].flatten()
    datetimes = (days * 24 * 3600) + times
    condition_gnss = (datetimes/3600 >= 25) & (datetimes/3600 <= 40.9)
    time_GNSS = datetimes[condition_gnss]/3600
    lat = data_unit['lat'].flatten()[condition_gnss]
    lon = data_unit['lon'].flatten()[condition_gnss]
    elev = data_unit['elev'].flatten()[condition_gnss]

    data_DOG = sio.loadmat('../../data/DOG/DOG1-camp.mat')
    data_DOG = data_DOG["tags"].astype(float)
    acoustic_DOG = np.unwrap(data_DOG[:,1]/1e9*2*np.pi)/(2*np.pi)
    offset = 68056+65.1
    time_DOG = (data_DOG[:,0]+ offset)/3600
    condition_DOG = (time_DOG >= 25) & (time_DOG <= 40.9)
    time_DOG = time_DOG[condition_DOG]
    acoustic_DOG = acoustic_DOG[condition_DOG]

    # Créez la grille autour de xyzR
    grid_ecef = create_grid_around(xyzR)
    print(grid_ecef, grid_ecef.shape)
    # Convertir la grille ECEF à géodésique
    grid_geodetic = [ecef2geodetic(x, y, z) for x, y, z in grid_ecef]
    grid_geodetic = np.array(grid_geodetic)  # Convertir en tableau NumPy pour faciliter l'indexation
    print(grid_geodetic)
    t=time.time()
    differences=(calculate_differences(grid_geodetic, lat, lon, elev, time_GNSS, time_DOG, acoustic_DOG))
    print(differences)
    print(time.time()-t)
    lat_grid = grid_geodetic[:, 0]
    lon_grid = grid_geodetic[:, 1]
    elev_grid = grid_geodetic[:, 2]

    x_grid = grid_ecef[:, 0]
    y_grid = grid_ecef[:, 1]
    z_grid = grid_ecef[:, 2]

    xR, yR, zR = geodetic2ecef(xyzR[0], xyzR[1], xyzR[2])

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    sc1 = plt.scatter(x_grid, y_grid, c=differences, cmap='jet', s=100)  # s=50 pour augmenter la taille
    plt.scatter(xR, yR, label='Optimal point', s=20, alpha=0.7, edgecolors='black')  # s=20 pour réduire la taille, alpha pour transparence
    plt.colorbar(sc1).set_label('RMS (s)')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('RMS in X vs Y')

    # Scatter plot pour x vs z
    plt.subplot(1, 3, 2)
    sc2 = plt.scatter(x_grid, z_grid, c=differences, cmap='jet', s=100)  # s=50 pour augmenter la taille
    plt.scatter(xR, zR, label='Optimal point', s=20, alpha=0.7, edgecolors='black')  # s=20 pour réduire la taille, alpha pour transparence
    plt.colorbar(sc2).set_label('RMS (s)')
    plt.xlabel('X')
    plt.ylabel('Z')
    plt.title('RMS in X vs Z')

    # Scatter plot pour y vs z
    plt.subplot(1, 3, 3)
    sc3 = plt.scatter(y_grid, z_grid, c=differences, cmap='jet', s=100)  # s=50 pour augmenter la taille
    plt.scatter(yR, zR, label='Optimal point', s=20, alpha=0.7, edgecolors='black')  # s=20 pour réduire la taille, alpha pour transparence
    plt.colorbar(sc3).set_label('RMS (s)')
    plt.xlabel('Y')
    plt.ylabel('Z')
    plt.title('RMS in Y vs Z')

    plt.tight_layout()
    plt.legend()
    plt.savefig('RMS_1cm_1mm.png',dpi=500)
    plt.show()
