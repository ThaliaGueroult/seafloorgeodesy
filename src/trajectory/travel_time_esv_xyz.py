import os
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import scipy.io as sio
from pymap3d import geodetic2ecef, ecef2geodetic
import datetime

def calculate_intermediate_point(xyzS, xyzR):
    '''
    Inputs:
    xyzS - Source coordinates
    xyzR - Receiver coordinates

    Outputs:
    xyzI - Intermediate point coordinates
    '''
    xs, ys, zs = xyzS[0], xyzS[1], xyzS[2]
    xr, yr, zr = xyzR[0], xyzR[1], xyzR[2]

    # Calculate the distance between the source and receiver
    d = np.sqrt((xr - xs) ** 2 + (yr - ys) ** 2)

    # Calculate the coordinates of the intermediate point I
    xi = xs + d
    yi = ys
    zi = zr

    xyzI = [xi, yi, zi]
    print(xyzI)
    return xyzI

def xyz2dec(xyzS, xyzR):
    '''
    Inputs:
    xyzS
    xyzR

    Outputs:
    dz - Vertical distance between source and receiver
    beta - Elevation angle from source to receiver
    '''
    elevS=xyzS[2]
    elevR=xyzR[2]
    # Calculate the intermediate point coordinates
    xyzI = calculate_intermediate_point(xyzS, xyzR)

    xs, ys, zs = xyzS[0], xyzS[1], xyzS[2]
    xi, yi, zi = xyzI[0], xyzI[1], xyzI[2]

    # Calculate the vertical distance
    dz = elevR-elevS

    # Calculate the horizontal distance
    dx = xi - xs
    dy = yi - ys

    distance_horizontale = np.sqrt(dx**2 + dy**2)

    # Calculate the elevation angle
    beta = math.degrees(math.atan2(dz, distance_horizontale))

    return dz, beta


def find_esv(beta, dz):
    # Charger la matrice
    folder_path = '../esv/esv_table_without_tol/global_table_interp'
    matrix_file_path = os.path.join(folder_path, 'global_table_esv.mat')
    data = sio.loadmat(matrix_file_path)

    dz_values = data['distance'].flatten()
    angle_values = data['angle'].flatten()
    esv_matrix = data['matrice']

    # Trouvez l'indice le plus proche pour dz et beta
    idx_closest_dz = np.argmin(np.abs(dz_values - dz))
    idx_closest_beta = np.argmin(np.abs(angle_values - beta))

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
        print(beta,dz)

        if beta < 0 :
            beta = - beta

        esv = find_esv(beta, dz)

        # xs, ys, zs = geodetic2ecef(xyzS[0], xyzS[1], xyzS[2])
        # xr, yr, zr = geodetic2ecef(xyzR[0], xyzR[1], xyzR[2])
        xs, ys, zs = xyzS[0], xyzS[1], xyzS[2]
        xr, yr, zr = xyzR[0], xyzR[1], xyzR[2]

        travel_time = np.sqrt((xs-xr)**2+(ys-yr)**2+(zs-zr)**2)/ esv
        travel_time_cst = np.sqrt((xs-xr)**2+(ys-yr)**2+(zs-zr)**2)/1515
        # Append the travel time to the list
        travel_times.append(travel_time)
        travel_times_cst.append(travel_time_cst)
        diff.append(travel_time_cst - travel_time)

    return travel_times, travel_times_cst, diff

def filter_outliers_xyz(x, y, z, time):
    '''
    Filtre pour enlever les données z aberrantes

    Args:
    x (array)
    y (array)
    z (array)

    Returns:
    x_filt (array)
    y_filt (array)
    z_filt (array)
    '''
    # Calculer Q1 et Q3
    Q1 = np.percentile(z, 25)
    Q3 = np.percentile(z, 75)

    # Calculer l'IQR
    IQR = Q3 - Q1

    # Définir les seuils pour filtrer les valeurs aberrantes
    lower_threshold = Q1 - 1.5 * IQR
    upper_threshold = Q3 + 1.5 * IQR

    # Filtrer les valeurs aberrantes
    z_filt = z[(z >= lower_threshold) & (z <= upper_threshold)]
    # Filtrer les valeurs correspondantes dans lat et lon
    x_filt = x[(z >= lower_threshold) & (z <= upper_threshold)]
    y_filt = y[(z >= lower_threshold) & (z <= upper_threshold)]
    time_filt = time[(z >= lower_threshold) & (z <= upper_threshold)]

    trajectory = []

    for i in range (len(z_filt)):
        point = (x_filt[i], y_filt[i], z_filt[i])
        trajectory.append(point)

    return trajectory, time_filt


if __name__ == '__main__':
    #If we work in geodetic UTM
    xyzR=[1.977967e6, -5.073198e6, 3.3101016e6]

    data_unit = sio.loadmat('../../data/SwiftNav_Data/Unit2-camp_bis.mat')
    # Extraire les variables du fichier
    days = data_unit['days'].flatten()
    times = data_unit['times'].flatten()
    lat = data_unit['lat'].flatten()
    lon = data_unit['lon'].flatten()
    elev = data_unit['elev'].flatten()
    x = data_unit['x'].flatten()
    y = data_unit['y'].flatten()
    z = data_unit['z'].flatten()


    # Créer une figure avec deux sous-graphiques côte à côte
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

    # Premier sous-graphique : Visualisation de x, y, z
    scatter1 = ax1.scatter(x, y, c=z, cmap='viridis', s=10)
    cbar1 = plt.colorbar(scatter1, ax=ax1)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title('Visualisation de la trajectoire 2D (x, y, z)')
    cbar1.set_label('Z', rotation=270, labelpad=15)

    # Deuxième sous-graphique : Visualisation de lat, lon, elev
    scatter2 = ax2.scatter(lat, lon, c=elev, cmap='viridis', s=10)
    cbar2 = plt.colorbar(scatter2, ax=ax2)
    ax2.set_xlabel('Latitude')
    ax2.set_ylabel('Longitude')
    ax2.set_title('Visualisation de la trajectoire 2D (lat, lon, elev)')
    cbar2.set_label('Elevation', rotation=270, labelpad=15)

    # Ajuster l'espacement entre les sous-graphiques
    plt.tight_layout()

    # Afficher la figure
    plt.show()

    print(x, y, z)

    days = days - 59015.

    datetimes = []
    for day, time in zip(days, times):
        # dt = datetime.timedelta(seconds=time) + day * 24 * 3600
        dt = time + day * 24 * 3600
        datetimes.append(dt)
    datetimes = np.array(datetimes)

    # Filtrer les valeurs aberrantes
    # traj_reel, datetime = filter_outliers(lat, lon, elev, datetimes)
    traj_reel, datetime = filter_outliers_xyz(x, y, z, datetimes)

    slant_range, slant_range_cst, diff = calculate_travel_times(traj_reel,xyzR)

    data = sio.loadmat('../../data/DOG/DOG1/DOG1-camp.mat')

    data = data["tags"].astype(float)
    plt.scatter((data[:,0]+68056)/3600,np.unwrap(data[:,1]/1e9*2*np.pi)/(2*np.pi), s = 1, label = 'Acoustic')
    print(data[0,0],datetime[0])
    plt.scatter((datetime)/3600, slant_range, s = 1 , label = 'SV GDEM Bermuda')
    plt.scatter((datetime)/3600, slant_range_cst, s = 1 , label = '1515 m/s')
    plt.ylabel('Time travel (s)')
    plt.xlabel('time [h]')
    plt.legend()
    plt.show()

    plt.figure()

    plt.scatter((datetime)/3600,slant_range_cst-slant_range, s = 1 , label = 'Travel time difference')
    plt.ylabel('Difference in travel_time (s)')
    plt.xlabel('time [h]')
    plt.legend()
    plt.show()
