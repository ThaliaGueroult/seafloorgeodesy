#!/usr/bin/python3
import os
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import scipy.io as sio
from pymap3d import geodetic2ecef

# Function to convert source and receiver coordinates to declination angle and elevation
def haversine_distance(lat1, lon1, lat2, lon2):
    # Convertir les latitudes et longitudes de degrés à radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # Rayon moyen de la Terre (en mètres)
    R = 6371e3

    # Calculer les différences entre les latitudes et longitudes
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Calculer la formule de haversine
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    # Calculer la distance
    return R * c

def xyz2dec(xyzS, xyzR):
    '''
    Inputs:
        xyzS - Source coordinates [longitude, latitude, altitude]
        xyzR - Receiver coordinates [longitude, latitude, altitude]

    Outputs:
        dz - Vertical distance between source and receiver (m)
        beta - Elevation angle from source to receiver (°)
    '''

    # Calculate the vertical distance
    dz = xyzR[2] - xyzS[:,2]
    dz = np.abs(dz)
    # Calculate horizontal distance using haversine formula
    dh = np.array([haversine_distance(lat_s, lon_s, xyzR[1], xyzR[0]) for lat_s, lon_s in zip(xyzS[:,1], xyzS[:,0])])
    # Calculate the elevation angle
    beta = np.degrees(np.arctan2(dz, dh))
    beta = np.abs(beta)  # Remove this line if you want to preserve negative angles when the source is above the receiver

    return beta, dz

# Function to find closest ESV value based on dz and beta
def find_esv(dz_array, angle_array, esv_matrix, beta, dz):
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

def GNSS_trajectory(lat, lon, elev):
    trajectory = list(zip(lat, lon, elev))
    return trajectory

def calculate_distances(trajectory, xyzR):
    ''' Calculate the distances from each point in the trajectory to the receiver. '''
    xr, yr, zr = geodetic2ecef(xyzR[0], xyzR[1], xyzR[2])
    xs, ys, zs = geodetic2ecef(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2])
    distances = np.sqrt((xs - xr)**2 + (ys - yr)**2 + (zs - zr)**2)
    return distances

def calculate_acoustic_distances(valid_acoustic_DOG, esv_matrix, dz, beta):
    ''' Calculate the distances using the acoustic times from the DOG multiplied by the ESV. '''
    # esv = find_esv(dz_array, angle_array, esv_matrix, beta, dz)
    acoustic_distances = valid_acoustic_DOG * 1515
    return acoustic_distances

def plot_distances(unit_name, time_GNSS, time_transducer, GNSS_distances, acoustic_distances,transducer_distances, svp, xyzR):
    # Plot setup
    label_text = f"{unit_name}, Lat: {xyzR[1]}, Lon: {xyzR[0]}, elev: {xyzR[2]}"
    plt.figure(figsize=(15, 10))
    plt.scatter(time_GNSS, GNSS_distances, s=10, label='GNSS estimation', alpha=1, marker='x', color='r', zorder=1)
    plt.scatter(time_GNSS, acoustic_distances, s=5, label='Acoustic data DOG', alpha=0.6, marker='o', color='b', zorder=2)
    plt.scatter(time_transducer, transducer_distances, s=8, label='Transducer estimation', alpha=1, marker='o', color='g', zorder=3)
    plt.ylabel('Distance (m)')
    plt.xlabel('Time [h]')
    plt.title(f'Distance Comparison of {unit_name} with {svp}')
    plt.text(min(time_GNSS), max(GNSS_distances), label_text, bbox=dict(facecolor='yellow', alpha=0.8))
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_difference(time, difference, title="Difference between GNSS and Transducer Distances"):
    plt.figure(figsize=(15, 10))
    plt.plot(time, difference, 'b-', label='Difference (GNSS - Transducer)')
    plt.xlabel('Time [h]')
    plt.ylabel('Difference (m)')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_combined(common_times, GNSS_distances, acoustic_distances, transducer_distances, difference, yaw_deg, time_yaw):
    plt.figure(figsize=(15, 20))

    # Plot difference between travel times
    plt.subplot(4, 1, 1)
    plt.plot(common_times, GNSS_distances - acoustic_distances, 'r-', label='Diff (GNSS - DOG)')
    plt.plot(common_times, transducer_distances - acoustic_distances, 'b-', label='Diff (Transducer - DOG)')
    plt.ylabel('Time Difference (ms)')
    plt.title('Difference in Travel Times')
    plt.legend()
    plt.grid(True)

    # Plotting the distances
    plt.subplot(4, 1, 2)
    plt.plot(common_times, GNSS_distances, 'r-', label='GNSS Distances')
    plt.plot(common_times, acoustic_distances, 'g-', label='Acoustic Distances (DOG)')
    plt.plot(common_times, transducer_distances, 'b-', label='Transducer Distances')
    plt.ylabel('Distance (m)')
    plt.title('Distances from the Different Methods')
    plt.legend()
    plt.grid(True)

    # Plot difference between GNSS and Transducer distances
    plt.subplot(4, 1, 3)

    # Masks for positive and negative values
    positive_diff = difference >= 0
    negative_diff = difference < 0

    plt.scatter(common_times[positive_diff], difference[positive_diff], s = 2, c = 'g')
    plt.scatter(common_times[negative_diff], difference[negative_diff], s = 2, c = 'r')
    plt.xlabel('Times (h)')
    plt.ylabel('Distance Difference (m)')
    plt.title('Difference in Distances between the two pairs DOG/GNSS and DOG/transducer')
    plt.legend()
    plt.grid(True)

    # Plot Yaw
    plt.subplot(4, 1, 4)
    plt.plot(time_yaw, yaw_deg, 'c-', label='Yaw (degrees)')
    plt.xlabel('Times (h)')
    plt.ylabel('Yaw (degrees)')
    plt.title('Yaw over Time')
    plt.legend()
    plt.grid(True)
    plt.subplots_adjust(hspace=0.5)  # Adjust the space between subplots
    plt.show()


if __name__ == '__main__':
    # Load ESV matrix data
    folder_path = '../esv/castbermuda/global_table_interp'
    matrix_file_path = os.path.join(folder_path, 'global_table_esv.mat')
    data = sio.loadmat(matrix_file_path)
    dz_array = data['distance'].flatten()
    angle_array = data['angle'].flatten()
    esv_matrix = data['matrice']


    # Load GNSS trajectory data
    unit_number = 1  # Change this for each unit or transducer
    mat_file_path = f'../../data/SwiftNav_Data/Unit{unit_number}-camp_bis.mat'
    data_unit = sio.loadmat(mat_file_path)
    days = data_unit['days'].flatten() - 59015
    times = data_unit['times'].flatten()
    datetimes = (days * 24 * 3600) + times
    condition_gnss = (datetimes / 3600 >= 25) & (datetimes / 3600 <= 37)
    time_GNSS = datetimes[condition_gnss] / 3600
    lat_filtered = data_unit['lat'].flatten()[condition_gnss]
    lon_filtered = data_unit['lon'].flatten()[condition_gnss]
    elev_filtered = data_unit['elev'].flatten()[condition_gnss]
    traj_reel = np.column_stack([lat_filtered, lon_filtered, elev_filtered])

    # Load DOG data and apply conditions
    data_DOG = sio.loadmat('../../data/DOG/DOG1-camp.mat')['tags']
    offset = 68121
    acoustic_DOG = np.unwrap(data_DOG[:, 1] / 1e9 * 2 * np.pi) / (2 * np.pi)
    time_DOG = (data_DOG[:, 0] + offset) / 3600
    condition_DOG = (time_DOG >= 25) & (time_DOG <= 37)
    time_DOG, acoustic_DOG = time_DOG[condition_DOG], acoustic_DOG[condition_DOG]

    # Initialize array to hold valid DOG data matching GNSS timestamps
    valid_acoustic_DOG = np.full(time_GNSS.shape, np.nan)
    common_indices = np.isin(time_GNSS, time_DOG)
    indices_in_DOG = np.searchsorted(time_DOG, time_GNSS[common_indices])
    valid_acoustic_DOG[common_indices] = acoustic_DOG[indices_in_DOG]

    data_transducer = sio.loadmat('data_transducer_ecef.mat')
    # data_transducer = sio.loadmat('transducer_positions.mat')
    time_transducer = data_transducer['times'].flatten()
    condition_transducer = (time_transducer >= 25) & (time_transducer <= 37)
    lat_transducer = data_transducer['lat'].flatten()[condition_transducer]
    lon_transducer = data_transducer['lon'].flatten()[condition_transducer]
    elev_transducer = data_transducer['elev'].flatten()[condition_transducer]
    time_transducer = time_transducer[condition_transducer]


    traj_transducer = np.column_stack([lat_transducer, lon_transducer, elev_transducer])

    print(len(lat_transducer),len(time_transducer))

    # Load xyzR
    xyzR = [31.46358041, -68.70136571, -5286.21569638]
    # xyzR = [31.46356091,   291.29859266, -5271.47395559]  # Use the correct xyzR for each unit

    # Calculate GNSS-based distances
    GNSS_distances = calculate_distances(traj_reel, xyzR)
    transducer_distances = calculate_distances(traj_transducer, xyzR)

    # Calculate beta and dz for the ESV lookup
    beta, dz = xyz2dec(traj_reel, xyzR)

    # Calculate acoustic-based distances using DOG times
    acoustic_distances = calculate_acoustic_distances(valid_acoustic_DOG, esv_matrix, dz, beta)

    # Plot the data
    plot_distances("Transducer", time_GNSS, time_transducer, GNSS_distances, acoustic_distances, transducer_distances, 'castbermuda', xyzR)

    common_times, GNSS_indices, transducer_indices = np.intersect1d(time_GNSS, time_transducer, return_indices=True)

    common_GNSS_distances = GNSS_distances[GNSS_indices]

    common_transducer_distances = transducer_distances[transducer_indices]

    distance_difference = common_GNSS_distances - common_transducer_distances
    # Filtrage des distances et des temps selon les indices communs
    common_acoustic_distances = acoustic_distances[GNSS_indices]


    # Load the MATLAB file
    data = sio.loadmat('attitude_data.mat')

    # Extract the yaw data
    yaw_deg = data['yaw_deg'].flatten()  # .flatten() to ensure it's a 1D array
    time_yaw = data['common_datetimes'].flatten()

    plot_difference(common_times, distance_difference)
    plot_combined(common_times, common_GNSS_distances, common_acoustic_distances, common_transducer_distances, distance_difference, yaw_deg, time_yaw)
