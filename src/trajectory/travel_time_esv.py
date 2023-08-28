#!/usr/bin/python3
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

# Charger la matrice d'esv
folder_path = '../esv/esv_table_without_tol/global_table_interp'
matrix_file_path = os.path.join(folder_path, 'global_table_esv.mat')
data = sio.loadmat(matrix_file_path)

dz_array = data['distance'].flatten()
angle_array = data['angle'].flatten()
esv_matrix = data['matrice']

def calculate_intermediate_point(xyzS, xyzR):
    '''
    Inputs:
    xyzS - Source coordinates [longitude, latitude, altitude]
    xyzR - Receiver coordinates [longitude, latitude, altitude]

    Outputs:
    xyzI - Intermediate point coordinates [longitude, latitude, altitude]
    '''
    # Convert source and receiver coordinates from geodetic to ECEF
    xs, ys, zs = geodetic2ecef(xyzS[0], xyzS[1], xyzS[2])
    xr, yr, zr = geodetic2ecef(xyzR[0], xyzR[1], xyzR[2])

    # Calculate the distance between the source and receiver
    d = np.sqrt((xr - xs) ** 2 + (yr - ys) ** 2)

    # Calculate the coordinates of the intermediate point I
    xi = xs + d
    yi = ys
    zi = zr

    xyzI = [xi, yi, zr]

    return xyzI

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
    xi, yi, zi = xyzI[0], xyzI[1], xyzI[2]

    # Calculate the vertical distance
    dz = elevR-elevS

    # Calculate the horizontal distance
    dx = xi - xs
    dy = yi - ys
    distance_horizontale = np.sqrt(dx**2 + dy**2)

    # Calculate the elevation angle
    beta = beta = np.degrees(np.arctan2(dz, distance_horizontale))

    return dz, beta

def generate_trajectory():
    '''
    Generate a trajectory in the shape of an "8" with varying altitude.

    Returns:
    trajectory (list): List of tuples representing the trajectory points in the format (latitude, longitude, altitude).
    '''

    lat_center = 31.45
    lon_center = 291.30
    lat_radius = 0.1
    lon_radius = 0.1
    num_points = 10000  # Number of points on the trajectory

    trajectory = []

    for i in range(num_points):
        t = float(i) / (num_points - 1)  # Normalized value between 0 and 1
        angle = t * 2 * math.pi

        lat = lat_center + lat_radius * math.sin(angle)
        lon = lon_center + lon_radius * math.sin(2 * angle)
        elev = 5 * math.sin(angle)  # Sinusoidal altitude between -5 m and 5 m

        point = (lat, lon, elev)
        trajectory.append(point)

    return trajectory


def find_esv(dz_array, angle_array, esv_matrix, beta, dz):
    # Trouvez l'indice le plus proche pour dz et beta
    idx_closest_dz = np.argmin(np.abs(dz_array - dz))
    idx_closest_beta = np.argmin(np.abs(angle_array - beta))

    # Récupérer l'esv correspondant
    closest_esv = esv_matrix[idx_closest_dz, idx_closest_beta]
    return closest_esv


def calculate_travel_times(trajectory, receiver):
    '''
    Calculate the travel times from each point in the trajectory to the receiver.

    Args:
    trajectory (list): List of tuples representing the trajectory points in the format (latitude, longitude, altitude).
    receiver (tuple): Tuple representing the coordinates of the receiver in the format (latitude, longitude, elevation).
    dz (float): Vertical distance between receiver and each point in the trajectory.
    esv_func (function): A function that takes beta (elevation angle) and dz (vertical distance) as inputs and returns the ESV value.

    Returns:
    travel_times (list): List of travel times in seconds for each point in the trajectory.
    '''

    travel_times = []
    travel_times_cst = []
    diff = []
    xyzS = [0,0,0]

    # Loop through each point in the trajectory
    for point in trajectory:
        xyzS[0], xyzS[1], xyzS[2] = point

        beta, dz = xyz2dec(xyzS,xyzR)

        if beta < 0 :
            beta = - beta

        esv = find_esv(beta, dz)

        xs, ys, zs = geodetic2ecef(xyzS[0], xyzS[1], xyzS[2])
        xr, yr, zr = geodetic2ecef(xyzR[0], xyzR[1], xyzR[2])

        travel_time = np.sqrt((xs-xr)**2+(ys-yr)**2+(zs-zr)**2) / esv
        travel_time_cst = np.sqrt((xs-xr)**2+(ys-yr)**2+(zs-zr)**2) / 1500
        # Append the travel time to the list
        travel_times.append(travel_time)
        travel_times_cst.append(travel_time_cst)
        diff.append(travel_time_cst - travel_time)

    return travel_times, travel_times_cst, diff

def filter_outliers(lat, lon, elev, time):
    '''
    Filtre les valeurs aberrantes des données de latitude, longitude et élévation.

    Args:
    lat (array): Données de latitude.
    lon (array): Données de longitude.
    elev (array): Données d'élévation.

    Returns:
    lat_filt (array): Données de latitude filtrées.
    lon_filt (array): Données de longitude filtrées.
    elev_filt (array): Données d'élévation filtrées.
    '''
    # Calculer Q1 et Q3
    Q1 = np.percentile(elev, 25)
    Q3 = np.percentile(elev, 75)

    # Calculer l'IQR
    IQR = Q3 - Q1

    # Définir les seuils pour filtrer les valeurs aberrantes
    lower_threshold = Q1 - 1.5 * IQR
    upper_threshold = Q3 + 1.5 * IQR

    # Filtrer les valeurs aberrantes
    elev_filt = elev[(elev >= lower_threshold) & (elev <= upper_threshold)]
    # Filtrer les valeurs correspondantes dans lat et lon
    lat_filt = lat[(elev >= lower_threshold) & (elev <= upper_threshold)]
    lon_filt = lon[(elev >= lower_threshold) & (elev <= upper_threshold)]
    time_filt = time[(elev >= lower_threshold) & (elev <= upper_threshold)]

    trajectory = []

    for i in range (len(elev_filt)):
        point = (lat_filt[i], lon_filt[i], elev_filt[i])
        trajectory.append(point)

    return trajectory, time_filt


def calculate_travel_times_optimized(trajectory, xyzR):
    '''
    Calculate the travel times from each point in the trajectory to the receiver using optimization techniques.
    '''

    # Preallocate arrays for speed
    num_points = len(trajectory)
    travel_times = np.zeros(num_points)
    travel_times_cst = np.zeros(num_points)
    diff = np.zeros(num_points)

    # Convert xyzR once since it's constant
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

def GNSS_trajectory(lat, lon, elev):
    trajectory = list(zip(lat, lon, elev))
    return trajectory


def objective_function(xyzR, lat, lon, elev, acoustic_DOG, time_GNSS, time_DOG):
    traj_reel = GNSS_trajectory(lat, lon, elev)
    slant_range, _, _ = calculate_travel_times_optimized(traj_reel, xyzR)

    # Interpoler acoustic_DOG pour les horodatages time_GNSS
    interpolated_acoustic_DOG = np.interp(time_GNSS, time_DOG, acoustic_DOG)
    print(xyzR)
    # Calculer la somme des carrés des différences
    return np.sum((slant_range - interpolated_acoustic_DOG)**2)


if __name__ == '__main__':

    xyzR = [31.46, 291.31, 5275]
    xyzR = [  31.46357361,  291.29870318, 5187.34216509]
    xyzR = [  31.46357361,  291.29870318, 5187.3421122 ]
    #optimal guess
    xyzR = [31.46356378,  291.29858793, 5186.53046237]
    print(geodetic2ecef(31.46356378,  291.29858793, 5186.53046237))


    # data_unit = sio.loadmat('../../data/SwiftNav_Data/Unit3-camp_bis.mat')
    # data_unit = sio.loadmat('../../data/SwiftNav_Data/Average_GPStime_synchro.mat')
    data_unit = sio.loadmat('../../data/SwiftNav_Data/GPStime_synchro/Unit3-camp_bis_GPStime_synchro.mat')

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
    offset = 68056+65.1
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
