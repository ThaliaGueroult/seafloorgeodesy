import os
import numpy as np
import pandas as pd
import math
from geodesy_toolbox import geod2ecef
import matplotlib.pyplot as plt
import scipy.io as sio

def calculate_intermediate_point(xyzS, xyzR):
    '''
    Inputs:
    xyzS - Source coordinates [longitude, latitude, altitude]
    xyzR - Receiver coordinates [longitude, latitude, altitude]

    Outputs:
    xyzI - Intermediate point coordinates [longitude, latitude, altitude]
    '''
    # Convert source and receiver coordinates from geodetic to ECEF
    xs, ys, zs = geod2ecef(xyzS[0], xyzS[1], xyzS[2])
    xr, yr, zr = geod2ecef(xyzR[0], xyzR[1], xyzR[2])

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
    xs, ys, zs = geod2ecef(xyzS[0], xyzS[1], xyzS[2])
    xi, yi, zi = xyzI[0], xyzI[1], xyzI[2]

    # Calculate the vertical distance
    dz = -elevR-elevS

    # Calculate the horizontal distance
    dx = xi - xs
    dy = yi - ys
    distance_horizontale = np.sqrt(dx**2 + dy**2)

    # Calculate the elevation angle
    beta = math.degrees(math.atan2(dz, distance_horizontale))

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


def find_esv(beta, dz):
    # print(beta,dz)
    folder_path = '../esv/esv_table_without_tol'
    closest_file = None
    closest_dz = float('inf')

    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)

            # Extract dz from the file name
            current_dz = float(filename.split('_')[1])

            # Find the file with the dz closest to the given dz (considering 1-meter step)
            dz_diff = abs(current_dz - dz)
            if dz_diff < closest_dz:
                closest_dz = dz_diff
                closest_file = file_path

    if closest_file:
        # Load data from the file with the closest dz
        data = pd.read_csv(closest_file, header=None)
        angle = data[0].values
        esv = data[1].values


        idx_nearest_beta = np.argmin(np.abs(angle - beta))
        closest_esv = esv[idx_nearest_beta]

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
        # print (xyzS[2])

        beta, dz = xyz2dec(xyzS,xyzR)

        if beta < 0 :
            beta = - beta

        esv = find_esv(beta, dz)

        # print(beta,dz)

        xs, ys, zs = geod2ecef(xyzS[0], xyzS[1], xyzS[2])
        xr, yr, zr = geod2ecef(xyzR[0], xyzR[1], xyzR[2])

        travel_time = np.sqrt((xs-xr)**2+(ys-yr)**2+(zs-zr)**2)/ esv
        travel_time_cst = np.sqrt((xs-xr)**2+(ys-yr)**2+(zs-zr)**2)/1500
        # Append the travel time to the list
        travel_times.append(travel_time)
        travel_times_cst.append(travel_time_cst)
        diff.append(travel_time_cst - travel_time)

    return travel_times, travel_times_cst, diff

def filter_outliers(lat, lon, elev):
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

    trajectory = []

    for i in range (len(elev_filt)):
        point = (lat_filt[i], lon_filt[i], elev_filt[i])
        trajectory.append(point)

    return trajectory


if __name__ == '__main__':
    trajectory_8 = generate_trajectory()

    xyzR=[31.45, 291.13, 5225]

    # slant_range, slant_range_cst, diff = calculate_travel_times(trajectory_8,xyzR)
    #
    # plt.plot(diff)
    # plt.ylabel('Time travel difference (s)')
    # plt.xlabel('Point Index')
    # plt.show()

    data = sio.loadmat('../../data/SwiftNav_Data/Unit1-camp.mat')
    print("GNSS: ", data)

    data = sio.loadmat('../../data/DOG/DOG1/DOG1-camp.mat')
    print("\n\nBEACON: ",data)
    print("\n\nHEADER",data.keys())
    print(data["tags"])

    data = data["tags"].astype(float)

    new_data = np.zeros(data[:,1].shape)
    new_data[0] = data[0,1]
    k = 0
    for i in range(1, data.shape[0]):
        if data[i-1,1] - data[i,1] > 9/10*1000000000:
            k+=1
        else:
            new_data[i] = 1000000000*k + data[i,1]
    print(np.max(data[:,1]))
    print(np.min(data[:,1]))
    plt.figure()
    plt.scatter(data[:,0]/3600, data[:,1]/10**8,s=1)
    # plt.scatter(data[:,0]/3600, new_data,s=1)
    plt.show()

    # # Extraire les données
    # lat = data['d'][0]['lat'][0].flatten()
    # lon = data['d'][0]['lon'][0].flatten()
    # elev = data['d'][0]['height'][0].flatten()  # Flatten the elev numpy array
    #
    # # Filtrer les valeurs aberrantes
    # traj_reel = filter_outliers(lat, lon, elev)
    #
    # slant_range, slant_range_cst, diff = calculate_travel_times(traj_reel,xyzR)
    #
    # plt.plot(slant_range, label = 'SV GDEM Bermuda')
    # plt.plot(slant_range_cst, label = '1515 m/s')
    # plt.ylabel('Time travel (s)')
    # plt.xlabel('Point Index')
    # plt.legend()
    # plt.show()
    #
    # y=np.arange(0,len(slant_range),1)
    #
    # plt.scatter(slant_range,y, label = 'SV GDEM Bermuda')
    # plt.scatter(slant_range_cst,y, label = '1500 m/s')
    # plt.ylabel('Time travel (s)')
    # plt.xlabel('Point Index')
    # plt.legend()
    # plt.show()
    #
    # plt.plot(diff)
    # plt.ylabel('Time travel difference (s)')
    # plt.xlabel('Point Index')
    # plt.show()
