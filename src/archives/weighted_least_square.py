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

def weighted_error_function(variables, xa, ya, za, da, weights):
    x, y, z = variables
    distances = np.sqrt((xa - x)**2 + (ya - y)**2 + (za - z)**2)
    residuals = distances - da
    weighted_residuals = residuals**2 * weights
    return np.sum(weighted_residuals)

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
def filter_data_by_common_times(A, common_times):
    mask = np.isin(A[0], common_times)
    return A[0][mask], A[1][mask], A[2][mask], A[3][mask]


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
da1 = np.sqrt(16.6**2 + 2.46**2 + 15.24**2)
da2 = np.sqrt(16.6**2 + 7.39**2 + 15.24**2)
da3 = np.sqrt(6.40**2 + 9.57**2 + 15.24**2)
da4 = np.sqrt(6.40**2 + 2.46**2 + 15.24**2)
da = np.array([da1, da2, da3, da4])


weights = np.ones(4)  # Puisque vous avez une incertitude de 1 mètre pour chaque mesure
estimated_positions = np.zeros((len(common_times), 3))
residuals_list = np.zeros(len(common_times))

for i in tqdm(range(len(common_times)), desc="Processing datetimes"):
    x = np.array([A4_filtered[1][i], A4_filtered[2][i], A4_filtered[3][i]])

    xa = np.array([A1_filtered[1][i], A2_filtered[1][i], A3_filtered[1][i], A4_filtered[1][i]])
    ya = np.array([A1_filtered[2][i], A2_filtered[2][i], A3_filtered[2][i], A4_filtered[2][i]])
    za = np.array([A1_filtered[3][i], A2_filtered[3][i], A3_filtered[3][i], A4_filtered[3][i]])

    result = least_squares(weighted_error_function, x, args=(xa, ya, za, da, weights))

    estimated_positions[i, :] = result.x

    misfit = weighted_error_function(result.x, xa, ya, za, da, weights)
    # print(misfit)
    rms = np.sqrt(misfit/4)
    # print(f"Estimated position [x,y,z] for {int(datetimes[i]*100)/100}s :", result.x)
    print(f"RMSE : ", rms, "m\n")
