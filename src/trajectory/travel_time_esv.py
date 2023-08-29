#!/usr/bin/python3
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

# Function to calculate intermediate point between Source (S) and Receiver (R)
def calculate_intermediate_point(xyzS, xyzR):
    """
    This function calculates the intermediate point coordinates between the source and receiver.
    Input:
        xyzS: [lat, lon, alt] of Source
        xyzR: [lat, lon, alt] of Receiver
    Output:
        Intermediate point coordinates
    """
    # Convert lat, lon, alt to Earth-Centered Earth-Fixed (ECEF) coordinates
    xs, ys, zs = geodetic2ecef(xyzS[0], xyzS[1], xyzS[2])
    xr, yr, zr = geodetic2ecef(xyzR[0], xyzR[1], xyzR[2])

    # Calculate the distance between the source and receiver
    d = np.sqrt((xr - xs) ** 2 + (yr - ys) ** 2)

    # Calculate the coordinates of the intermediate point I
    xi = xs + d
    yi = ys
    zi = zr

    xyzI = ecef2geodetic(xi, yi, zr)

    return xyzI

# Function to convert source and receiver coordinates to declination angle and elevation
def xyz2dec(xyzS, xyzR):
    '''
    Inputs:
    xyzS - Source coordinates [longitude, latitude, altitude]
    xyzR - Receiver coordinates [longitude, latitude, altitude]

    Outputs:
    dz - Vertical distance between source and receiver
    beta - Elevation angle from source to receiver
    '''
    elevS=xyzS[2]
    elevR=xyzR[2]
    # Calculate the intermediate point coordinates
    xyzI = calculate_intermediate_point(xyzS, xyzR)
    xs, ys, zs = geodetic2ecef(xyzS[0], xyzS[1], xyzS[2])
    xi, yi, zi = geodetic2ecef(xyzI[0], xyzI[1], xyzI[2])

    # Calculate the vertical distance
    dz = elevR-elevS

    # Calculate the horizontal distance
    dx = xi - xs
    dy = yi - ys
    distance_horizontale = np.sqrt(dx**2 + dy**2)

    # Calculate the elevation angle
    beta = math.degrees(math.atan2(dz, distance_horizontale))

    return beta,dz

# Function to generate a sinusoidal trajectory for the source
def generate_trajectory():
    """
    This function generates a sinusoidal trajectory for the source.
    Output:
        Returns a DataFrame containing the trajectory
    """
    # Define parameters for the sinusoidal trajectory
    lat_center = 31.45
    lon_center = 291.30
    elev_center = 5000
    amplitude = 1000
    period = 360

    # Generate the trajectory
    time = np.linspace(0, 360, 361)
    lat_traj = lat_center + amplitude * np.sin(np.radians(time))
    lon_traj = lon_center + amplitude * np.sin(np.radians(time))
    elev_traj = elev_center + amplitude * np.sin(np.radians(time))

    # Create a DataFrame to hold the trajectory
    trajectory = pd.DataFrame({'Time': time, 'Latitude': lat_traj, 'Longitude': lon_traj, 'Elevation': elev_traj})

    return trajectory

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
    # Locate the closest dz and beta index
    idx_closest_dz = np.argmin(np.abs(dz_array - dz))
    idx_closest_beta = np.argmin(np.abs(angle_array - beta))

    # Récupérer l'esv correspondant
    closest_esv = esv_matrix[idx_closest_dz, idx_closest_beta]
    return closest_esv

#
def calculate_travel_times_optimized(trajectory, xyzR):
    '''
    Calculate the travel times from each point in the trajectory to the receiver.

    Args:
    trajectory (list): List of tuples representing the trajectory points in the format (latitude, longitude, altitude).
    receiver (tuple): Tuple representing the coordinates of the receiver in the format (latitude, longitude, elevation).
    Returns:
    travel_times (list): List of travel times in seconds for each point in the trajectory.
    '''

    # Preallocate arrays for speed
    num_points = len(trajectory)
    travel_times = np.zeros(num_points)
    travel_times_cst = np.zeros(num_points)
    diff = np.zeros(num_points)

    # # Convert xyzR from geodetic (lat,lon,elev) to ECEF
    xr, yr, zr = geodetic2ecef(xyzR[0], xyzR[1], xyzR[2])

    for i, point in enumerate(trajectory):
        beta, dz = xyz2dec(point, xyzR)

        if beta < 0:
            beta = -beta

        esv = find_esv(dz_array, angle_array, esv_matrix,beta, dz)
        xs, ys, zs = geodetic2ecef(point[0], point[1], point[2])

        travel_time = np.sqrt((xs-xr)**2 + (ys-yr)**2 + (zs-zr)**2) / esv
        travel_time_cst = np.sqrt((xs-xr)**2 + (ys-yr)**2 + (zs-zr)**2) / 1515

        travel_times[i] = travel_time
        travel_times_cst[i] = travel_time_cst
        diff[i] = travel_time_cst - travel_time

    return travel_times, travel_times_cst, diff
#
# def calculate_travel_times_optimized(trajectory, xyzR):
#     # Convert input to numpy arrays for vectorization
#     trajectory = np.array(trajectory)
#     xyzR = np.array(xyzR)
#
#     # # Convert xyzR once since it's constant
#     # xr, yr, zr = geodetic2ecef(xyzR[0], xyzR[1], xyzR[2])
#     #
#     # # Convert all trajectory points from geodetic to ECEF
#     # xs, ys, zs = geodetic2ecef(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2])
#
#     # Calculate beta, dz for all points
#     beta, dz = xyz2dec(trajectory, xyzR)
#
#     # Use absolute value for beta
#     beta = np.abs(beta)
#
#     # Find esv values for all points
#     esv = find_esv(dz_array, angle_array, esv_matrix, beta, dz)
#
#     # Calculate the repeated term
#     repeated_term = np.sqrt((xs - xr)**2 + (ys - yr)**2 + (zs - zr)**2)
#
#     # Calculate travel times
#     travel_times = repeated_term / esv
#     travel_times_cst = repeated_term / 1515
#
#     # Calculate differences
#     diff = travel_times_cst - travel_times
#
#     return travel_times, travel_times_cst, diff


def GNSS_trajectory(lat, lon, elev):
    trajectory = list(zip(lat, lon, elev))
    return trajectory

def objective_function(xyzR, lat, lon, elev, acoustic_DOG, time_GNSS, time_DOG):
    traj_reel = GNSS_trajectory(lat, lon, elev)
    slant_range, _, _ = calculate_travel_times_optimized(traj_reel, xyzR)

    # Interpoler acoustic_DOG pour les horodatages time_GNSS
    interpolated_acoustic_DOG = np.interp(time_GNSS, time_DOG, acoustic_DOG)

    # Calculer la somme des carrés des différences
    return np.sum((slant_range - interpolated_acoustic_DOG)**2)


if __name__ == '__main__':

    xyzR = [31.46, 291.31, 5275]
    #optimal guess
    xyzR = [31.46356378,  291.29858793, 5186.53046237]
    print(geodetic2ecef(31.46356378,  291.29858793, 5186.53046237))


    # data_unit = sio.loadmat('../../data/SwiftNav_Data/Unit1-camp_bis.mat') #+65
    data_unit = sio.loadmat('../../data/SwiftNav_Data/Average_GPStime_synchro.mat')  #+67
    # data_unit = sio.loadmat('../../data/SwiftNav_Data/GPStime_synchro/Unit1-camp_bis_GPStime_synchro.mat') #+65

    days = data_unit['days'].flatten() - 59015
    times = data_unit['times'].flatten()
    datetimes = (days * 24 * 3600) + times
    condition_gnss = (datetimes/3600 >= 25) & (datetimes/3600 <= 40.9)
    time_GNSS = datetimes[condition_gnss]/3600

    lat = data_unit['lat'].flatten()
    lon = data_unit['lon'].flatten()
    elev = data_unit['elev'].flatten()
    lat = lat[condition_gnss]
    lon = lon[condition_gnss]
    elev = elev[condition_gnss]
    traj_reel = GNSS_trajectory(lat, lon, elev)
    slant_range, slant_range_cst, difference = calculate_travel_times_optimized(traj_reel, xyzR)
    data_DOG = sio.loadmat('../../data/DOG/DOG1-camp.mat')
    data_DOG = data_DOG["tags"].astype(float)
    acoustic_DOG = np.unwrap(data_DOG[:,1]/1e9*2*np.pi)/(2*np.pi)
    offset = 68056+68
    time_DOG = (data_DOG[:,0]+ offset)/3600
    condition_DOG = ((data_DOG[:,0] + offset)/3600 >= 25) & ((data_DOG[:,0] + offset)/3600 <= 40.9)
    time_DOG = time_DOG[condition_DOG]
    acoustic_DOG = acoustic_DOG[condition_DOG]

    # Supprime les 'nan' de time_DOG et acoustic_DOG
    valid_DOG_indices = ~np.isnan(time_DOG) & ~np.isnan(acoustic_DOG)
    time_DOG_clean = time_DOG[valid_DOG_indices]
    acoustic_DOG_clean = acoustic_DOG[valid_DOG_indices]

    # Supprime les 'nan' de time_GNSS
    valid_GNSS_indices = ~np.isnan(time_GNSS)
    time_GNSS_clean = time_GNSS[valid_GNSS_indices]

    # Interpoler acoustic_DOG_clean sur time_GNSS_clean
    interpolated_acoustic_DOG = np.interp(time_GNSS_clean, time_DOG_clean, acoustic_DOG_clean)

    # Calculer la différence pour ces indices valides
    difference_data = slant_range[valid_GNSS_indices] - interpolated_acoustic_DOG

    valid_data_indices = ~np.isnan(slant_range[valid_GNSS_indices])
    difference_data = slant_range[valid_GNSS_indices][valid_data_indices] - interpolated_acoustic_DOG[valid_data_indices]
    print(np.sqrt(np.mean(difference_data**2)))




    # plt.scatter(time_GNSS, difference_data, s = 1, label = 'Difference (slant_range - acoustic_DOG)')

    # plt.ylabel('Time travel (s)')
    # plt.xlabel('time [h]')
    # plt.xlim(25, 41)
    # plt.legend()
    # plt.show()


    # Première figure
    plt.figure(figsize=(10, 8))  # Ajustez la taille de la fenêtre si nécessaire

    # Graphique du haut (Acoustic vs SV GDEM Bermuda)
    plt.subplot(2, 1, 1)  # 2 lignes, 1 colonne, 1ère figure
    plt.scatter(time_DOG, acoustic_DOG, s = 3, label = 'Acoustic')
    plt.scatter(time_GNSS, slant_range, s = 1 , label = 'SV GDEM Bermuda')
    plt.ylabel('Time travel (s)')
    plt.xlabel('time [h]')
    plt.xlim(25, 41)
    plt.legend()

    # Graphique du bas (Différence)
    plt.subplot(2, 1, 2)  # 2 lignes, 1 colonne, 2ème figure
    interpolated_acoustic_DOG = np.interp(time_GNSS, time_DOG, acoustic_DOG)
    difference_data = slant_range - interpolated_acoustic_DOG
    print(np.sqrt(np.mean(difference_data**2)))
    plt.scatter(time_GNSS, difference_data, s = 1, label = 'Difference (slant_range - acoustic_DOG)')
    plt.ylabel('Time travel (s)')
    plt.xlabel('time [h]')
    plt.xlim(25, 41)
    plt.legend()

    # Afficher les graphiques
    plt.tight_layout()  # Assure que les étiquettes et les titres ne se chevauchent pas
    plt.show()



    # '''Pour le troisième dog'''
    # xyzR = [31.45, 291.32, 4000]
    # xyzR = [31.44955433,  291.30765693, 3162.54807501]
    # data_unit = sio.loadmat('../../data/SwiftNav_Data/Unit1-camp_bis.mat')
    #
    # days = data_unit['days'].flatten() - 59015
    # times = data_unit['times'].flatten()
    # datetimes = (days * 24 * 3600) + times
    # condition_gnss = (datetimes/3600 >= 25) & (datetimes/3600 <= 41)
    # time_GNSS = datetimes[condition_gnss]/3600
    #
    # lat = data_unit['lat'].flatten()
    # lon = data_unit['lon'].flatten()
    # elev = data_unit['elev'].flatten()
    # lat = lat[condition_gnss]
    # lon = lon[condition_gnss]
    # elev = elev[condition_gnss]
    # traj_reel = GNSS_trajectory(lat, lon, elev)
    #
    # slant_range, slant_range_cst, difference = calculate_travel_times_optimized(traj_reel, xyzR)
    #
    # data_DOG = sio.loadmat('../../data/DOG/DOG3-camp.mat')
    # data_DOG = data_DOG["tags"].astype(float)
    # acoustic_DOG = np.unwrap(data_DOG[:,1]/1e9*2*np.pi)/(2*np.pi)
    # offset3 = 68625-1700
    # time_DOG = (data_DOG[:,0]+offset3)/3600
    # condition_DOG = ((data_DOG[:,0]+offset3)/3600 >= 25) & ((data_DOG[:,0]+offset3)/3600 <= 41)
    # time_DOG = time_DOG[condition_DOG]
    # acoustic_DOG = acoustic_DOG[condition_DOG]
    #
    # plt.scatter(time_DOG, acoustic_DOG, s = 1, label = 'Acoustic')
    # plt.scatter(time_GNSS, slant_range, s = 1 , label = 'SV GDEM Bermuda')
    # # plt.scatter(time_GNSS, slant_range_cst, s = 1 , label = '1500 m/s')
    # plt.ylabel('Time travel (s)')
    # plt.xlabel('time [h]')
    # plt.legend()
    # plt.show()
    #
    #
    # # initial_guess = [31.45, 291.32, 5225]
    # # result = minimize(objective_function, initial_guess, args=(lat, lon, elev, acoustic_DOG, time_GNSS, time_DOG))
    # #
    # # optimal_xyzR = result.x
    # # print("Optimal xyzR:", optimal_xyzR)
