#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from pymap3d import geodetic2ecef, ecef2geodetic
from travel_time_esv import *

def create_grid_around(coord, extension_cm=1):
    step = 0.001  # 1 cm
    range_vals = np.arange(-extension_cm/100, extension_cm/100 + step, step)

    # Créez une grille 3D
    xx, yy, zz = np.meshgrid(range_vals, range_vals, range_vals, indexing='ij')

    # Ajustez la grille par rapport à 'coord'
    xx += coord[0]
    yy += coord[1]
    zz += coord[2]

    # Reshape pour avoir une liste de coordonnées
    grid_points = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))

    return grid_points

def calculate_difference_for_point(xyzR, lat, lon, elev, time_GNSS, time_DOG, acoustic_DOG):
    traj_reel = GNSS_trajectory([lat], [lon], [elev])
    slant_range, _, _ = calculate_travel_times_optimized(traj_reel, xyzR)
    interpolated_acoustic_DOG = np.interp(time_GNSS, time_DOG, acoustic_DOG)
    difference_data = slant_range - interpolated_acoustic_DOG
    return np.sqrt(np.mean(difference_data**2))

if __name__ == '__main__':
    # Vos données et fonctions préliminaires
    xyzR = [31.46356378,  291.29858793, 5186.53046237]

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
    print(grid_ecef)

    # Convertir la grille ECEF à géodésique
    grid_geodetic = [ecef2geodetic(x, y, z) for x, y, z in grid_ecef]

    differences = np.zeros(len(grid_geodetic))
    for i, (grid_lat, grid_lon, grid_elev) in enumerate(grid_geodetic):
        print([grid_lat, grid_lon, grid_elev])
        diff = calculate_difference_for_point([grid_lat, grid_lon, grid_elev], lat, lon, elev, time_GNSS, time_DOG, acoustic_DOG)
        differences[i] = diff

    # Afficher les graphiques
    grid = np.array(grid_ecef)
    differences = np.array(differences)

    # Graphe x,y
    plt.scatter(grid[:,0], grid[:,1], c=differences)
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Écart x,y")
    plt.show()

    # Graphe y,z
    plt.scatter(grid[:,1], grid[:,2], c=differences)
    plt.colorbar()
    plt.xlabel("y")
    plt.ylabel("z")
    plt.title("Écart y,z")
    plt.show()

    # Graphe x,z
    plt.scatter(grid[:,0], grid[:,2], c=differences)
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("z")
    plt.title("Écart x,z")
    plt.show()
