#!/usr/bin/python3
# Import required libraries

# Import required libraries
import os
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import scipy.io as sio
from pymap3d import geodetic2ecef, ecef2geodetic
import datetime
import time
from scipy.optimize import minimize
import numpy as np
import math
from scipy.optimize import least_squares

# Load the Effective Sound Velocity (ESV) matrix from a .mat file
folder_path = '../esv/castbermuda/global_table_interp'
matrix_file_path = os.path.join(folder_path, 'global_table_esv.mat')

# Load the matrix data and convert to float64
data = sio.loadmat(matrix_file_path)
dz_array = data['distance'].flatten().astype(np.float64)
angle_array = data['angle'].flatten().astype(np.float64)
esv_matrix = data['matrice'].astype(np.float64)

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

def calculate_travel_times_optimized(trajectory, xyzR):
        '''
        Calculate the travel times from each point in the trajectory to the receiver.

        Inputs :
            Trajectory : List of tuples representing the trajectory points in the format (latitude, longitude, altitude).
            xyzR : Receiver coordinates [longitude, latitude, altitude]
        Outputs
            travel_times (list): List of travel times in seconds for each point in the trajectory using ESV table
            travel_times_cst (list) : List of travel times in seconds for each point in the trajectory using a constant velocity
            diff (list) : diff between travel_time and travel_time_cst
        '''
        num_points = len(trajectory)
        trajectory = np.array(trajectory)

        beta, dz = xyz2dec(trajectory, xyzR)
        xr, yr, zr = geodetic2ecef(xyzR[0], xyzR[1], xyzR[2])
        xs, ys, zs = geodetic2ecef(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2])
        esv = find_esv(dz_array, angle_array, esv_matrix, beta, dz)  # Update this function to handle arrays

        distances_squared = (xs - xr)**2 + (ys - yr)**2 + (zs - zr)**2
        travel_times = np.sqrt(distances_squared) / esv
        travel_times_cst = np.sqrt(distances_squared) / 1515

        diff = travel_times_cst - travel_times

        return travel_times, travel_times_cst, diff

def GNSS_trajectory(lat, lon, elev):
    trajectory = list(zip(lat, lon, elev))
    return trajectory

def cost_function(xyzR, traj_reel, valid_acoustic_DOG, time_GNSS):
    slant_range, _, _ = calculate_travel_times_optimized(traj_reel, xyzR)

    # Calculez les résidus
    residuals = slant_range - valid_acoustic_DOG
    residuals = residuals[~np.isnan(residuals)]
    return residuals

if __name__ == '__main__':

    xyzR = np.array( [31.46347837,   -68.70145524, -5276.879054  ]).astype(np.float64)
    data_transduceur = sio.loadmat('transducer_pos(1).mat')

    datetimes = data_transduceur['datetimes'].flatten()
    condition_gnss = (datetimes >= 25) & (datetimes <= 38)
    time_GNSS = datetimes[condition_gnss]

    x = data_transduceur['x'].flatten()
    y = data_transduceur['y'].flatten()
    z = data_transduceur['z'].flatten()
    lat, lon, elev = ecef2geodetic(x, y, z)
    lat = lat[condition_gnss]
    lon = lon[condition_gnss]
    elev = elev[condition_gnss]
    print('elev',elev)

    traj_reel = GNSS_trajectory(lat, lon, elev)
    slant_range, slant_range_cst, difference = calculate_travel_times_optimized(traj_reel, xyzR)
    data_DOG = sio.loadmat('../../data/DOG/DOG1-camp.mat')
    data_DOG = data_DOG["tags"].astype(float)
    acoustic_DOG = np.unwrap(data_DOG[:,1]/1e9*2*np.pi)/(2*np.pi)
    offset =  68130
    time_DOG = (data_DOG[:,0]+ offset)/3600
    condition_DOG = ((data_DOG[:,0] + offset)/3600 >= 25) & ((data_DOG[:,0] + offset)/3600 <= 38)
    time_DOG = time_DOG[condition_DOG]
    acoustic_DOG = acoustic_DOG[condition_DOG]

    # Create an array of 'nan' values the same length as time_GNSS
    valid_acoustic_DOG = np.full(time_GNSS.shape, np.nan)

    # Find indices where time_GNSS and time_DOG have matching timestamps
    common_indices = np.isin(time_GNSS, time_DOG)

    # Find the corresponding indices in time_DOG for the common times
    indices_in_DOG = np.searchsorted(time_DOG, time_GNSS[common_indices])

    # Fill valid_acoustic_DOG array with the values of acoustic_DOG at the common timestamps
    valid_acoustic_DOG[common_indices] = acoustic_DOG[indices_in_DOG]


    difference_data = slant_range - valid_acoustic_DOG
    RMS = np.sqrt(np.nanmean(difference_data**2))
    print('\n RMS{} s'.format(RMS))


    # Create the main figure and axes
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), gridspec_kw={'width_ratios': [1, 4], 'height_ratios': [4, 1]}, sharey='row')
    label_text = f"Lat: {xyzR[1]}, Lon: {xyzR[0]}, elev: {xyzR[2]}"
    # Top plot (Acoustic vs SV GDEM Bermuda)
    axes[0, 1].scatter(time_DOG, acoustic_DOG, s=3, label='Acoustic DOG')
    axes[0, 1].scatter(time_GNSS, slant_range, s=1, label='GNSS SV Bermuda')
    axes[0, 1].set_ylabel('Time Travel (s)')
    axes[0, 1].set_xlabel('Time [h]')
    axes[0, 1].set_xlim(25, 39.5)
    axes[0, 1].text(25, max(acoustic_DOG), label_text)  # This puts the label at x=25 and y=max(acoustic_DOG)
    axes[0, 1].legend()

    # Bottom plot (Difference)
    axes[1, 1].scatter(time_GNSS, difference_data, s=1, label='Difference (slant_range - matched_acoustic_DOG)')
    axes[1, 1].set_xlabel('Time [h]')
    axes[1, 1].set_xlim(25, 39.5)
    axes[1, 1].text(25,10,'RMS = {} s'.format(RMS))
    axes[1, 1].legend()

    # Histogram
    axes[1, 0].hist(difference_data, orientation='horizontal', bins=30, alpha=0.5)
    axes[1, 0].set_xlabel('Frequency')
    axes[1, 0].set_title('Histogram')


    # Invert the x-axis to have bars going from right to left
    axes[1, 0].invert_xaxis()

    # Share y-axis for histogram and difference plot
    axes[1, 0].get_shared_y_axes().join(axes[1, 0], axes[1, 1])

    # Hide unnecessary axes
    axes[0, 0].axis('off')

    # Display the plots
    plt.tight_layout()
    plt.show()

    print(cost_function(xyzR, traj_reel, valid_acoustic_DOG, time_GNSS))
    result = least_squares(cost_function, xyzR, args=(traj_reel, valid_acoustic_DOG, time_GNSS), xtol = 1e-15)

    optimal_xyzR = result.x
    print(result)
    print("\n Optimal xyzR:", optimal_xyzR)
    residuals = cost_function(result.x, traj_reel, valid_acoustic_DOG, time_GNSS)
    rmse = np.sqrt(np.mean(residuals**2))

    print("RMSE:", rmse)
