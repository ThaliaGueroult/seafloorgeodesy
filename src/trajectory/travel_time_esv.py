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
from scipy.optimize import least_squares
from scipy.signal import find_peaks
from multiprocessing import Pool, cpu_count
import matplotlib.gridspec as gridspec

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

def calculate_travel_times_optimized(trajectory, xyzR, sv = 1515):
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
        travel_times_cst = np.sqrt(distances_squared) / sv
        diff = travel_times_cst - travel_times

        return travel_times, travel_times_cst, diff

def GNSS_trajectory(lat, lon, elev):
    trajectory = list(zip(lat, lon, elev))
    return trajectory

def objective_function(xyzR, traj_reel, valid_acoustic_DOG, time_GNSS):
    # Generate the new slant_range based on current xyzR
    slant_range, _, _ = calculate_travel_times_optimized(traj_reel, xyzR)

    # Calculate the difference
    difference_data = slant_range - valid_acoustic_DOG

    # Calculate the sum of squares of the differences
    result = np.sqrt(np.nanmean(difference_data**2))

    # Print the result for debugging
    print("Result:", result, xyzR)

    return result


def plot_unit_data(unit_number, svp):
    xyzR_list = [
        [31.46356091,   291.29859266, -5271.47395559],  # Unit 1
        [31.46355736,  291.29860692, -5271.16197333],  # Unit 2
        [31.46356016,   291.29861564, -5271.02494184],  # Unit 3
        [31.4635864,    291.2986076,  -5270.75686706],  # Unit 4
        [31.4635699,    291.29860731, -5270.79420961]]  # Unit 5, average

    #Add the offsets for the DOG synchronisation, computed manually by eye-shooting
    offsets = [68121, 68122, 68126, 68126, 68124]

    # Fetch unit-specific configurations
    xyzR = xyzR_list[unit_number - 1]
    offset = offsets[unit_number - 1]

    print(f'Processing Unit {unit_number}')

    # Load GNSS data
    mat_file_path = f'../../data/SwiftNav_Data/Unit{unit_number}-camp_bis.mat'
    data_unit = sio.loadmat(mat_file_path)
    days = data_unit['days'].flatten() - 59015
    times = data_unit['times'].flatten()
    datetimes = (days * 24 * 3600) + times
    condition_gnss = (datetimes / 3600 >= 25) & (datetimes / 3600 <= 37)
    time_GNSS = datetimes[condition_gnss] / 3600
    lat, lon, elev = data_unit['lat'].flatten()[condition_gnss], data_unit['lon'].flatten()[condition_gnss], data_unit['elev'].flatten()[condition_gnss]

    # Loading GNSS trajectory and computing slant_range time
    traj_reel = GNSS_trajectory(lat, lon, elev)
    slant_range, slant_range_cst , _ = calculate_travel_times_optimized(traj_reel, xyzR)

    # Load DOG data and apply conditions
    data_DOG = sio.loadmat('../../data/DOG/DOG1-camp.mat')['tags'].astype(float)
    acoustic_DOG = np.unwrap(data_DOG[:, 1] / 1e9 * 2 * np.pi) / (2 * np.pi)
    time_DOG = (data_DOG[:, 0] + offset) / 3600
    condition_DOG = (time_DOG >= 25) & (time_DOG <= 37)
    time_DOG, acoustic_DOG = time_DOG[condition_DOG], acoustic_DOG[condition_DOG]

    # Initialize array to hold valid DOG data matching GNSS timestamps
    valid_acoustic_DOG = np.full(time_GNSS.shape, np.nan)
    common_indices = np.isin(time_GNSS, time_DOG)
    indices_in_DOG = np.searchsorted(time_DOG, time_GNSS[common_indices])
    valid_acoustic_DOG[common_indices] = acoustic_DOG[indices_in_DOG]

    # Calculate differences and RMS
    difference_data = slant_range - valid_acoustic_DOG
    time_diff = time_GNSS[~np.isnan(difference_data)]
    difference_data = difference_data[~np.isnan(difference_data)]

    RMS = np.sqrt(np.nanmean(difference_data ** 2))
    print(f'\n RMS: {RMS*1e3} ms')


    specified_times = [25.401, 26.1709, 26.2772, 28.137, 29.4495, 30.5291, 31.8132, 33.12, 34.85, 36.52]
  # Prepare label and plot
    label_text = f"Antenna: {unit_number}, Lat: {xyzR[1]}, Lon: {xyzR[0]}, elev: {xyzR[2]}"
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), gridspec_kw={'width_ratios': [1, 4], 'height_ratios': [4, 1]})
    if svp == 'castbermuda':
        if unit_number in [1,2,3,4]:
            fig.suptitle(f'Comparison of GNSS estimation and raw DOG data, Antenna {unit_number}, with Bermuda SVP cast (2020)', y=0.92)
        else :
            fig.suptitle(f'Comparison of GNSS estimation and raw DOG data, Barycenter of antennas, with Bermuda SVP cast (2020)', y=0.92)
    else:
        if unit_number in [1,2,3,4]:
            fig.suptitle(f'Comparison of GNSS estimation and raw DOG data, Antenna {unit_number}, with GDEM SVP model (2003)', y=0.92)
        else :
            fig.suptitle(f'Comparison of GNSS estimation and raw DOG data, Barycenter of antennas, with GDEM SVP model (2003)', y=0.92)

    # Acoustic vs GNSS plot

    axes[0, 1].scatter(time_DOG, acoustic_DOG, s=5, label='Acoustic data DOG', alpha=0.6, marker='o', color='b', zorder=2)
    axes[0, 1].scatter(time_GNSS, slant_range, s=10, label='GNSS estimation', alpha=1, marker='x', color='r', zorder=1)
    axes[0, 1].set_ylabel('Travel Time (s)')
    axes[0, 1].set_xlabel('Time [h]')
    axes[0, 1].text(25, max(acoustic_DOG), label_text, bbox=dict(facecolor='yellow', alpha=0.8))
    axes[0, 1].legend()

    # Difference plot
    axes[1, 1].scatter(time_diff, difference_data*1e3, s=1)
    axes[1, 1].set_ylim(-np.max(np.abs(difference_data))*1e3*0.9, np.max(np.abs(difference_data))*1e3*1.10)
    axes[1, 1].set_xlabel('Time [h]')
    axes[1, 1].set_title('Difference between acoustic Data and GNSS estimation')

    # Histogram
    axes[1, 0].hist(difference_data*1e3, orientation='horizontal', bins=30, alpha=0.5)
    axes[1, 0].set_ylabel('Difference (ms)')
    axes[1, 0].set_xlabel('Frequency')
    axes[1, 0].set_title(f'RMSE : {format(RMS*1e3, ".4f")} ms')
    axes[1, 0].set_ylim(-np.max(np.abs(difference_data))*1e3*0.9, np.max(np.abs(difference_data))*1e3*1.10)
    axes[1, 0].invert_xaxis()
    axes[0, 0].axis('off')

    for time in specified_times:
        axes[0, 1].axvline(x=time, color='g', linestyle='--', alpha=0.7)
        axes[1, 1].axvline(x=time, color='g', linestyle='--', alpha=0.7)

    plt.show()
    filename = f"Comparison_Antenna_{unit_number}_{svp}.png"
    fig.savefig(filename, bbox_inches='tight')
    print(f"Figure saved as {filename}")

    return (xyzR, traj_reel, valid_acoustic_DOG, time_GNSS, difference_data)

def plot_unit_data_trans(method):

    if method == 'trilateration':
        xyzR =[31.46360178, -68.70133641, -5286.36159637]
        mat_file_path = 'transducer_positions_bis.mat'

    if method == 'attitude':
        xyzR = [31.46357068,  -68.70142218, -5286.31387099]
        mat_file_path = 'data_transducer_ecef.mat'

    if method == 'attitude_mean':
        mat_file_path = 'kabsch_average_ned.mat'
        xyzR = [31.4634716,    -68.70131008, -5277.88414035]




    #Add the offsets for the DOG synchronisation, computed manually by eye-shooting
    offset = [68125]

    print(f'Processing Transduceur')

    # mat_file_path = 'transducer_positions.mat'

    data_unit = sio.loadmat(mat_file_path)
    datetimes = data_unit['times'].flatten()
    condition_gnss = (datetimes >= 25) & (datetimes <= 37)
    time_GNSS = datetimes[condition_gnss]
    lat, lon, elev = data_unit['lat'].flatten()[condition_gnss], data_unit['lon'].flatten()[condition_gnss], data_unit['elev'].flatten()[condition_gnss]

    # Loading GNSS trajectory and computing slant_range time
    traj_reel = GNSS_trajectory(lat, lon, elev)
    slant_range, slant_range_cst, _ = calculate_travel_times_optimized(traj_reel, xyzR)

    # Load DOG data and apply conditions
    data_DOG = sio.loadmat('../../data/DOG/DOG1-camp.mat')['tags'].astype(float)
    acoustic_DOG = np.unwrap(data_DOG[:, 1] / 1e9 * 2 * np.pi) / (2 * np.pi)
    time_DOG = (data_DOG[:, 0] + offset) / 3600
    condition_DOG = (time_DOG >= 25) & (time_DOG <= 37)
    time_DOG, acoustic_DOG = time_DOG[condition_DOG], acoustic_DOG[condition_DOG]

    # Initialize array to hold valid DOG data matching GNSS timestamps
    valid_acoustic_DOG = np.full(time_GNSS.shape, np.nan)
    common_indices = np.isin(time_GNSS, time_DOG)
    indices_in_DOG = np.searchsorted(time_DOG, time_GNSS[common_indices])
    valid_acoustic_DOG[common_indices] = acoustic_DOG[indices_in_DOG]

    # Calculate differences and RMS
    difference_data = slant_range - valid_acoustic_DOG
    time_diff = time_GNSS[~np.isnan(difference_data)]
    difference_data = difference_data[~np.isnan(difference_data)]
    RMS = np.sqrt(np.nanmean(difference_data ** 2))
    print(f'\n RMS: {RMS*1e3} ms')

  # Prepare label and plot
    label_text = f"Transducer, Lat: {xyzR[1]}, Lon: {xyzR[0]}, elev: {xyzR[2]}"
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), gridspec_kw={'width_ratios': [1, 4], 'height_ratios': [1, 8]})
    fig.suptitle(f'Comparison of GNSS estimation and raw DOG data, Transducer estimated position, with Bermuda SVP cast (2020)', y=0.92)
    #
    # # Acoustic vs GNSS plot
    # axes[0, 1].scatter(time_DOG, acoustic_DOG, s=5, label='Acoustic data DOG', alpha=0.6, marker='o', color='b', zorder=2)
    # axes[0, 1].scatter(time_GNSS, slant_range, s=10, label='GNSS estimation', alpha=1, marker='x', color='r', zorder=1)
    # axes[0, 1].set_ylabel('Travel Time (s)')
    # axes[0, 1].set_xlabel('Time [h]')
    axes[1, 1].text(25, max(acoustic_DOG), label_text, bbox=dict(facecolor='yellow', alpha=0.8))
    # axes[0, 1].legend()
    axes[0,1].axis('off')

    # Difference plot
    axes[1, 1].scatter(time_diff, difference_data*1e3, s=1)
    axes[1, 1].set_xlabel('Time [h]')
    axes[1, 1].set_title('Difference between acoustic Data and GNSS estimation')
    axes[1, 1].set_ylim(-13, 13)


    # Histogram
    axes[1, 0].hist(difference_data*1e3, orientation='horizontal', bins=30, alpha=0.5)
    axes[1, 0].set_ylabel('Difference (ms)')
    axes[1, 0].set_xlabel('Frequency')
    axes[1, 0].set_title(f'RMSE : {format(RMS*1e3, ".4f")} ms')
    axes[1, 0].set_ylim(-13,13)
    axes[1, 0].invert_xaxis()
    axes[0, 0].axis('off')

    specified_times = [25.401, 26.1709, 26.2772, 28.137, 29.4495, 30.5291, 31.8132, 33.12, 34.85, 36.52]
    for time in specified_times:
        # axes[0, 1].axvline(x=time, color='g', linestyle='--', alpha=0.7)
        axes[1, 1].axvline(x=time, color='g', linestyle='--', alpha=0.7)

    plt.show()
    filename = f"Comparison_transducer.png"
    fig.savefig(filename, bbox_inches='tight')
    print(f"Figure saved as {filename}")

    return (xyzR, traj_reel, valid_acoustic_DOG, time_GNSS, difference_data)

def optimize_position(xyzR_initial, traj_reel, valid_acoustic_DOG, time_GNSS):
    result = least_squares(cost_function, xyzR_initial, args=(traj_reel, valid_acoustic_DOG, time_GNSS), xtol=1e-12)

    optimal_xyzR = result.x
    residuals = cost_function(result.x, traj_reel, valid_acoustic_DOG, time_GNSS)
    rmse = np.sqrt(np.mean(residuals**2))

    return optimal_xyzR, rmse

def optimize_plot_trans(filename):
    xyzR = [31.46356916, -68.70142096, -5287.39730994]
    offset = [68125]

    data_unit = sio.loadmat(filename)
    datetimes = data_unit['times'].flatten()
    condition_gnss = (datetimes >= 25) & (datetimes <= 37)
    time_GNSS = datetimes[condition_gnss]
    lat, lon, elev = data_unit['lat'].flatten()[condition_gnss], data_unit['lon'].flatten()[condition_gnss], data_unit['elev'].flatten()[condition_gnss]

    # Loading GNSS trajectory and computing slant_range time
    traj_reel = GNSS_trajectory(lat, lon, elev)
    # Load DOG data and apply conditions
    data_DOG = sio.loadmat('../../data/DOG/DOG1-camp.mat')['tags'].astype(float)
    acoustic_DOG = np.unwrap(data_DOG[:, 1] / 1e9 * 2 * np.pi) / (2 * np.pi)
    time_DOG = (data_DOG[:, 0] + offset) / 3600
    condition_DOG = (time_DOG >= 25) & (time_DOG <= 37)
    time_DOG, acoustic_DOG = time_DOG[condition_DOG], acoustic_DOG[condition_DOG]

    # Initialize array to hold valid DOG data matching GNSS timestamps
    valid_acoustic_DOG = np.full(time_GNSS.shape, np.nan)
    common_indices = np.isin(time_GNSS, time_DOG)
    indices_in_DOG = np.searchsorted(time_DOG, time_GNSS[common_indices])
    valid_acoustic_DOG[common_indices] = acoustic_DOG[indices_in_DOG]

    slant_range, slant_range_cst, _ = calculate_travel_times_optimized(traj_reel, xyzR)
    # Calculate differences and RMS
    difference_data = slant_range - valid_acoustic_DOG
    time_diff = time_GNSS[~np.isnan(difference_data)]
    difference_data = difference_data[~np.isnan(difference_data)]
    RMS = np.sqrt(np.nanmean(difference_data ** 2))

    optimal_xyzR, rmse = optimize_position(xyzR, traj_reel, valid_acoustic_DOG, time_GNSS)
    slant_range, slant_range_cst, _ = calculate_travel_times_optimized(traj_reel, xyzR)
    difference_data = slant_range - valid_acoustic_DOG
    time_diff = time_GNSS[~np.isnan(difference_data)]
    difference_data = difference_data[~np.isnan(difference_data)]
    RMS = np.sqrt(np.nanmean(difference_data ** 2))
    print(f'\n RMS: {RMS*1e3} ms')

    label_text = f"Transducer, Lat: {xyzR[1]}, Lon: {xyzR[0]}, elev: {xyzR[2]}"
    # Créez la grille de spécification pour vos sous-figures
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 4])  # Le deuxième graphique est 4 fois plus large que le premier

    # Créez la figure et les axes en utilisant cette grille
    fig = plt.figure(figsize=(18, 4))
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
    ylim_max = np.ceil(1.1 * np.max(np.abs(difference_data*1e3)))
    # Maintenant, vous pouvez utiliser ax0 et ax1 comme vos axes[0] et axes[1]
    fig.suptitle(f'Comparison of GNSS estimation and raw DOG data, Transducer estimated position, with Bermuda SVP cast (2020)')
    # Histogram (à gauche)
    ax0.hist(difference_data * 1e3, orientation='horizontal', bins=30, alpha=0.5)
    ax0.set_ylabel('Difference (ms)')
    ax0.set_xlabel('Frequency')
    ax0.set_title(f'RMSE : {format(RMS*1e3, ".4f")} ms')
    ax0.set_ylim(-ylim_max, ylim_max)

    # Résidus (à droite)
    ax1.scatter(time_diff, difference_data * 1e3, s=1)
    ax1.set_xlabel('Time [h]')
    ax1.set_title('Difference between acoustic Data and GNSS estimation')
    ax1.annotate(label_text, xy=(0, 1), xycoords='axes fraction', xytext=(5, -5), textcoords='offset points', bbox=dict(facecolor='yellow', alpha=0.8), ha='left', va='top')
    ax1.set_ylim(-ylim_max, ylim_max)
    specified_times = [25.401, 26.1709, 26.2772, 28.137, 29.4495, 30.5291, 31.8132, 33.12, 34.85, 36.52]
    for time in specified_times:
        ax1.axvline(x=time, color='g', linestyle='--', alpha=0.7)

    output_directory = os.path.join('levier_variations', 'img')
    base_name = os.path.basename(filename)  # Récupère le nom du fichier avec extension
    file_without_extension = os.path.splitext(base_name)[0]  # Récupère le nom du fichier sans extension
    figure_filename = file_without_extension + ".png"  # Ajoute l'extension .png

    # Quand vous voulez sauvegarder la figure :
    full_figure_path = os.path.join(output_directory, figure_filename)
    fig.savefig(full_figure_path, bbox_inches='tight')
    print(f"Figure saved as {full_figure_path}")
    return (optimal_xyzR, rmse)


def plot_unit_data_cst(unit_number, sv):
    xyzR_list = [
        [31.46356091,   291.29859266, -5271.47395559],  # Unit 1
        [31.46355736,  291.29860692, -5271.16197333],  # Unit 2
        [31.46356016,   291.29861564, -5271.02494184],  # Unit 3
        [31.4635864,    291.2986076,  -5270.75686706],  # Unit 4
        [31.4635699,    291.29860731, -5270.79420961]] # Unit 5, barcycenter

    #Add the offsets for the DOG synchronisation, computed manually by eye-shooting
    offsets = [68121, 68122, 68126, 68126, 68124]

    # Fetch unit-specific configurations
    xyzR = xyzR_list[unit_number - 1]
    offset = offsets[unit_number - 1]

    print(f'Processing Unit {unit_number}')

    # Load GNSS data
    mat_file_path = f'../../data/SwiftNav_Data/Unit{unit_number}-camp_bis.mat'
    data_unit = sio.loadmat(mat_file_path)
    days = data_unit['days'].flatten() - 59015
    times = data_unit['times'].flatten()
    datetimes = (days * 24 * 3600) + times
    condition_gnss = (datetimes / 3600 >= 25) & (datetimes / 3600 <= 37)
    time_GNSS = datetimes[condition_gnss] / 3600
    lat, lon, elev = data_unit['lat'].flatten()[condition_gnss], data_unit['lon'].flatten()[condition_gnss], data_unit['elev'].flatten()[condition_gnss]

    # Loading GNSS trajectory and computing slant_range time
    traj_reel = GNSS_trajectory(lat, lon, elev)
    _ , slant_range, _ = calculate_travel_times_optimized(traj_reel, xyzR, sv)

    # Load DOG data and apply conditions
    data_DOG = sio.loadmat('../../data/DOG/DOG1-camp.mat')['tags'].astype(float)
    acoustic_DOG = np.unwrap(data_DOG[:, 1] / 1e9 * 2 * np.pi) / (2 * np.pi)
    time_DOG = (data_DOG[:, 0] + offset) / 3600
    condition_DOG = (time_DOG >= 25) & (time_DOG <= 37)
    time_DOG, acoustic_DOG = time_DOG[condition_DOG], acoustic_DOG[condition_DOG]

    # Initialize array to hold valid DOG data matching GNSS timestamps
    valid_acoustic_DOG = np.full(time_GNSS.shape, np.nan)
    common_indices = np.isin(time_GNSS, time_DOG)
    indices_in_DOG = np.searchsorted(time_DOG, time_GNSS[common_indices])
    valid_acoustic_DOG[common_indices] = acoustic_DOG[indices_in_DOG]

    # Calculate differences and RMS
    difference_data = slant_range - valid_acoustic_DOG
    time_diff = time_GNSS[~np.isnan(difference_data)]
    difference_data = difference_data[~np.isnan(difference_data)]

    RMS = np.sqrt(np.nanmean(difference_data ** 2))
    print(f'\n RMS: {RMS*1e3} ms')

  # Prepare label and plot
    label_text = f"Antenna: {unit_number}, Lat: {xyzR[1]}, Lon: {xyzR[0]}, elev: {xyzR[2]}"
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), gridspec_kw={'width_ratios': [1, 4], 'height_ratios': [4, 1]})
    if unit_number in [1,2,3,4]:
        fig.suptitle(f'Comparison of GNSS estimation and raw DOG data, Antenna {unit_number}, with c = {sv} m/s', y=0.92)
    else :
        fig.suptitle(f'Comparison of GNSS estimation and raw DOG data, Barycenter of antennas, with c = {sv} m/s', y=0.92)

    # Acoustic vs GNSS plot

    axes[0, 1].scatter(time_DOG, acoustic_DOG, s=5, label='Acoustic data DOG', alpha=0.6, marker='o', color='b', zorder=2)
    axes[0, 1].scatter(time_GNSS, slant_range, s=10, label='GNSS estimation', alpha=1, marker='x', color='r', zorder=1)
    axes[0, 1].set_ylabel('Travel Time (s)')
    axes[0, 1].set_xlabel('Time [h]')
    axes[0, 1].text(25, max(acoustic_DOG), label_text, bbox=dict(facecolor='yellow', alpha=0.8))
    axes[0, 1].legend()

    # Difference plot
    axes[1, 1].scatter(time_diff, difference_data*1e3, s=1)
    axes[1, 1].set_ylim(-np.max(np.abs(difference_data))*1e3*0.9, np.max(np.abs(difference_data))*1e3*1.10)
    axes[1, 1].set_xlabel('Time [h]')
    axes[1, 1].set_title('Difference between acoustic Data and GNSS estimation')

    # Histogram
    axes[1, 0].hist(difference_data*1e3, orientation='horizontal', bins=30, alpha=0.5)
    axes[1, 0].set_ylabel('Difference (ms)')
    axes[1, 0].set_xlabel('Frequency')
    axes[1, 0].set_title(f'RMSE : {format(RMS*1e3, ".4f")} ms')
    axes[1, 0].set_ylim(-np.max(np.abs(difference_data))*1e3*0.9, np.max(np.abs(difference_data))*1e3*1.10)
    axes[1, 0].invert_xaxis()
    axes[0, 0].axis('off')

    specified_times = [25.401, 26.1709, 26.2772, 28.137, 29.4495, 30.5291, 31.8132, 33.12, 34.85, 36.52]
    for time in specified_times:
        axes[0, 1].axvline(x=time, color='g', linestyle='--', alpha=0.7)
        axes[1, 1].axvline(x=time, color='g', linestyle='--', alpha=0.7)

    filename = f"Comparison_Antenna_{unit_number}_{sv}ms.png"
    fig.savefig(filename, bbox_inches='tight')
    print(f"Figure saved as {filename}")

    return (xyzR, traj_reel, valid_acoustic_DOG, time_GNSS, difference_data)

def cost_function(xyzR, traj_reel, valid_acoustic_DOG, time_GNSS):
    slant_range, _, _ = calculate_travel_times_optimized(traj_reel, xyzR)

    # Calculez les résidus
    residuals = slant_range - valid_acoustic_DOG
    residuals = residuals[~np.isnan(residuals)]
    return residuals

def main():
    leviers_variations_path = 'levier_variations'
    mat_files = [f for f in os.listdir(leviers_variations_path) if f.endswith('.mat')]

    # Créez un pool de processus, utilisez autant de processus que votre CPU le permet
    with Pool(cpu_count()) as pool:
        # Utilisez le map pour appliquer la fonction à chaque fichier
        results = pool.map(optimize_plot_trans, [os.path.join(leviers_variations_path, mat_file) for mat_file in mat_files])

    # Affichez les résultats si vous le souhaitez
    for res in results:
        print(res)

if __name__ == '__main__':

    # Load the Effective Sound Velocity (ESV) matrix from a .mat file
    # Path to the folder containing the .mat file
    folder_path = '../esv/castbermuda/global_table_interp'
    # folder_path = '../esv/castbermuda/global_table_interp'
    matrix_file_path = os.path.join(folder_path, 'global_table_esv.mat')
    data = sio.loadmat(matrix_file_path)
    dz_array = data['distance'].flatten()
    angle_array = data['angle'].flatten()
    esv_matrix = data['matrice']

    # Chemin vers le dossier contenant les fichiers .mat
    leviers_variations_path = "levier_variations"

    # Liste de tous les fichiers .mat dans le dossier
    mat_files = [f for f in os.listdir(leviers_variations_path) if f.endswith('.mat')]

    for mat_file in mat_files:
        # Chemin complet vers le fichier .mat actuel
        full_path = os.path.join(leviers_variations_path, mat_file)

        # Appel à la fonction
        results = optimize_plot_trans(full_path)
        print(results)

    # # Exemple d'utilisation
    # for i in range (1,6):
        # xyzR, traj_reel, valid_acoustic_DOG, time_GNSS, difference_data = plot_unit_data(i, 'GDEM')
        # xyzR, traj_reel, valid_acoustic_DOG, time_GNSS, difference_data = plot_unit_data_cst(i,1516)
    # xyzR, traj_reel, valid_acoustic_DOG, time_GNSS, difference_data = plot_unit_data_trans('attitude_mean')
    # xyzR, traj_reel, valid_acoustic_DOG, time_GNSS, difference_data = plot_unit_data_cst(4,1516)
    # xyzR, traj_reel, valid_acoustic_DOG, time_GNSS, difference_data = plot_unit_data(5, 'GDEM')
    # # xyzR, traj_reel, valid_acoustic_DOG, time_GNSS, difference_data = plot_unit_data(5, 'castbermuda')
    # print(cost_function(xyzR, traj_reel, valid_acoustic_DOG, time_GNSS))
    # result = least_squares(cost_function, xyzR, args=(traj_reel, valid_acoustic_DOG, time_GNSS), xtol = 1e-12)
    #
    # optimal_xyzR = result.x
    # print(result)
    # print("\n Optimal xyzR:", optimal_xyzR)
    # residuals = cost_function(result.x, traj_reel, valid_acoustic_DOG, time_GNSS)
    # rmse = np.sqrt(np.mean(residuals**2))
    #
    # print("RMSE:", rmse)

    # # xyzR_list = [
    # #     [31.46356091,   291.29859266, -5271.47395559],  # Unit 1
    # #     [31.46355736,  291.29860692, -5271.16197333],  # Unit 2
    # #     [31.46356016,   291.29861564, -5271.02494184],  # Unit 3
    # #     [31.4635864,    291.2986076,  -5270.75686706],  # Unit 4
    # #     [31.4635699,    291.29860731, -5270.79420961]]  # Unit 5, average
    # #
    # # Coordonnées géodésiques pour les deux points
    # lat1, lon1, elev1 =   31.46360178, -68.70133641, -5286.36159637
    # lat2, lon2, elev2 = 31.46357068,  -68.70142218, -5286.31387099
    #
    # # Conversion en coordonnées ECEF
    # x1, y1, z1 = geodetic2ecef(lat1, lon1, elev1)
    # x2, y2, z2 = geodetic2ecef(lat2, lon2, elev2)
    #
    # # Calcul de la distance euclidienne
    # distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
    #
    # print(f"La distance entre les deux points est {distance:.2f} mètres.")
    #
    # # initial_guess = [31.46, 291.29, -5265]
    # # result = minimize(objective_function, initial_guess, args=(traj_reel, valid_acoustic_DOG, time_GNSS))
    # #
    # # optimal_xyzR = result.x
    # # print(result)
    # # print("\n Optimal xyzR:", optimal_xyzR)
    #
    # # plt.ioff()
