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

# Load the Effective Sound Velocity (ESV) matrix from a .mat file
# Path to the folder containing the .mat file
folder_path = '../esv/castbermuda/global_table_interp'
matrix_file_path = os.path.join(folder_path, 'global_table_esv.mat')

# Load the matrix data
data = sio.loadmat(matrix_file_path)

# Flatten the arrays for easier manipulation
dz_array = data['distance'].flatten()
angle_array = data['angle'].flatten()
esv_matrix = data['matrice']
print(esv_matrix.shape)

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
    xs, ys, zs = geodetic2ecef(xyzS[:,0], xyzS[:,1], xyzS[:,2])
    xr, yr, zr = geodetic2ecef(xyzR[0], xyzR[1], xyzR[2])

    # Calculate the vertical distance
    dz = xyzR[2] - xyzS[:,2]
    dx = xr - xs
    dy = yr - ys

    dh = np.sqrt(dx**2 + dy**2)

    # Calculate the elevation angle
    beta = np.degrees(np.arctan2(dz, dh))
    beta = np.abs(beta)
    return beta,dz

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

def objective_function(xyzR, traj_reel, valid_acoustic_DOG, time_GNSS):
    # Generate the new slant_range based on current xyzR
    slant_range, _, _ = calculate_travel_times_optimized(traj_reel, xyzR)

    # Calculate the difference
    difference_data = slant_range - valid_acoustic_DOG

    # Calculate the sum of squares of the differences
    result = np.sqrt(np.nanmean(difference_data**2))

    # Print the result for debugging
    print("Result:", result)

    return result

def plot_unit_data(unit_number):
    # Initialize unit-specific configurations
    xyzR_list = [
        [31.46356973, 291.29859242, 5191.15569819],  # Unit 1
        [31.46356189, 291.29859139, 5190.50899803],  # Unit 2
        [31.46356844, 291.29858342, 5189.94309155],  # Unit 3
        [31.46357847, 291.29858509, 5190.81298335],   # Unit 4
        [31.46356976, 291.29858848, 5190.53628934]   # Unit 5, average
    ]
    offsets = [68121, 68121, 68126, 68126, 68124]

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
    condition_gnss = (datetimes / 3600 >= 25) & (datetimes / 3600 <= 40.9)
    time_GNSS = datetimes[condition_gnss] / 3600
    lat, lon, elev = data_unit['lat'].flatten()[condition_gnss], data_unit['lon'].flatten()[condition_gnss], data_unit['elev'].flatten()[condition_gnss]

    # Loading GNSS trajectory and computing slant_range time
    traj_reel = GNSS_trajectory(lat, lon, elev)
    slant_range, _, _ = calculate_travel_times_optimized(traj_reel, xyzR)

    # Load DOG data and apply conditions
    data_DOG = sio.loadmat('../../data/DOG/DOG3-camp.mat')['tags'].astype(float)
    acoustic_DOG = np.unwrap(data_DOG[:, 1] / 1e9 * 2 * np.pi) / (2 * np.pi)
    time_DOG = (data_DOG[:, 0] + offset) / 3600
    condition_DOG = (time_DOG >= 25) & (time_DOG <= 40.9)
    time_DOG, acoustic_DOG = time_DOG[condition_DOG], acoustic_DOG[condition_DOG]

    # Initialize array to hold valid DOG data matching GNSS timestamps
    valid_acoustic_DOG = np.full(time_GNSS.shape, np.nan)
    common_indices = np.isin(time_GNSS, time_DOG)
    indices_in_DOG = np.searchsorted(time_DOG, time_GNSS[common_indices])
    valid_acoustic_DOG[common_indices] = acoustic_DOG[indices_in_DOG]

    # Calculate differences and RMS
    difference_data = slant_range - valid_acoustic_DOG
    RMS = np.sqrt(np.nanmean(difference_data ** 2))
    print(f'\n RMS: {RMS} s')

  # Prepare label and plot
    label_text = f"Antenna: {unit_number}, Lat: {xyzR[1]}, Lon: {xyzR[0]}, elev: {xyzR[2]}, RMS: {RMS} s"
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), gridspec_kw={'width_ratios': [1, 4], 'height_ratios': [4, 1]})
    fig.suptitle(f'Comparison of GNSS estimation and raw DOG data, Antenna {unit_number}', y=0.92)

    # Acoustic vs GNSS plot

    axes[0, 1].scatter(time_DOG, acoustic_DOG, s=5, label='Acoustic data DOG', alpha=0.6, marker='o', color='b', zorder=2)
    axes[0, 1].scatter(time_GNSS, slant_range, s=10, label='GNSS estimation', alpha=1, marker='x', color='r', zorder=1)
    axes[0, 1].set_ylabel('Travel Time (s)')
    axes[0, 1].set_xlabel('Time [h]')
    axes[0, 1].text(25, max(acoustic_DOG), label_text, bbox=dict(facecolor='yellow', alpha=0.8))
    axes[0, 1].legend()

    # Difference plot
    axes[1, 1].scatter(time_GNSS, difference_data, s=1)
    axes[1, 1].set_xlabel('Time [h]')
    axes[1, 1].set_title('Difference between acoustic Data and GNSS estimation')
    axes[1, 1].legend()

    # Histogram
    axes[1, 0].hist(difference_data, orientation='horizontal', bins=30, alpha=0.5)
    axes[1, 0].set_ylabel('Difference (s)')
    axes[1, 0].set_xlabel('Frequency')
    axes[1, 0].invert_xaxis()

    axes[0, 0].axis('off')

    plt.show()

def plot_unit_data_elev(unit_number):
    # Initialize unit-specific configurations
    xyzR_list = [
        [31.46356973, 291.29859242, 5191.15569819],  # Unit 1
        [31.46356189, 291.29859139, 5190.50899803],  # Unit 2
        [31.46356844, 291.29858342, 5189.94309155],  # Unit 3
        [31.46357847, 291.29858509, 5190.81298335],   # Unit 4
        [31.46356976, 291.29858848, 5190.53628934]   # Unit 5, average
    ]
    offsets = [68121, 68121, 68126, 68126, 68124]

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
    condition_gnss = (datetimes / 3600 >=27.056) & (datetimes / 3600 <= 27.066)
    time_GNSS = datetimes[condition_gnss] / 3600
    lat, lon, elev = data_unit['lat'].flatten()[condition_gnss], data_unit['lon'].flatten()[condition_gnss], data_unit['elev'].flatten()[condition_gnss]

    # Loading GNSS trajectory and computing slant_range time
    traj_reel = GNSS_trajectory(lat, lon, elev)
    slant_range, _, _ = calculate_travel_times_optimized(traj_reel, xyzR)

    # Load DOG data and apply conditions
    data_DOG = sio.loadmat('../../data/DOG/DOG1-camp.mat')['tags'].astype(float)
    acoustic_DOG = np.unwrap(data_DOG[:, 1] / 1e9 * 2 * np.pi) / (2 * np.pi)
    time_DOG = (data_DOG[:, 0] + offset) / 3600
    condition_DOG = (time_DOG >= 27.056) & (time_DOG <= 27.066)
    time_DOG, acoustic_DOG = time_DOG[condition_DOG], acoustic_DOG[condition_DOG]

    # Initialize array to hold valid DOG data matching GNSS timestamps
    valid_acoustic_DOG = np.full(time_GNSS.shape, np.nan)
    common_indices = np.isin(time_GNSS, time_DOG)
    indices_in_DOG = np.searchsorted(time_DOG, time_GNSS[common_indices])
    valid_acoustic_DOG[common_indices] = acoustic_DOG[indices_in_DOG]

    # Calculate differences and RMS
    difference_data = slant_range - valid_acoustic_DOG
    RMS = np.sqrt(np.nanmean(difference_data ** 2))
    print(f'\n RMS: {RMS} s')

  # Prepare label and plot
    label_text = f"Antenna: {unit_number}, Lat: {xyzR[1]}, Lon: {xyzR[0]}, elev: {xyzR[2]}, RMS: {RMS} s"
    fig, axes = plt.subplots(3, 2, figsize=(15, 15), gridspec_kw={'width_ratios': [1, 4], 'height_ratios': [4, 1, 4]})
    fig.suptitle(f'Comparison of GNSS estimation and raw DOG data, Antenna {unit_number}', y=0.92)

    # Acoustic vs GNSS plot
    axes[0, 1].scatter(time_DOG, acoustic_DOG, s=20, label='Acoustic data DOG', alpha=0.6, marker='o', color='b', zorder=2)
    axes[0, 1].scatter(time_GNSS, slant_range, s=30, label='GNSS estimation', alpha=1, marker='x', color='r', zorder=1)
    axes[0, 1].set_ylabel('Travel Time (s)')
    axes[0, 1].set_xlabel('Time [h]')
    axes[0, 1].text(25, max(acoustic_DOG), label_text, bbox=dict(facecolor='yellow', alpha=0.8))
    axes[0, 1].legend()

    # Difference plot
    axes[1, 1].scatter(time_GNSS, difference_data*1e3, s=30)
    axes[1, 1].set_xlabel('Time [h]')
    axes[1, 1].set_title('Difference between acoustic Data and GNSS estimation')
    axes[1, 1].legend()

    # Histogram
    axes[1, 0].hist(difference_data*1e3, orientation='horizontal', bins=30, alpha=0.5)
    axes[1, 0].set_ylabel('Difference (ms)')
    axes[1, 0].set_xlabel('Frequency')
    axes[1, 0].invert_xaxis()

    # New elevation plot
    axes[2, 1].scatter(time_GNSS, elev, s=30, label='Elevation data', alpha=0.6, marker='o', color='g')
    axes[2, 1].set_ylabel('Elevation (m)')
    axes[2, 1].set_xlabel('Time [h]')
    axes[2, 1].legend()

    # Turn off unused axes
    axes[0, 0].axis('off')
    axes[2, 0].axis('off')

    plt.show()


if __name__ == '__main__':

    xyzR = [  31.46361704,  291.29957503, 5231.14237279] #GNSS4

    data_transduceur = sio.loadmat('data_transduceur.mat') #+70


    datetimes = data_transduceur['times'].flatten()
    condition_gnss = (datetimes >= 25) & (datetimes <= 38.5)
    time_GNSS = datetimes[condition_gnss]

    x = data_transduceur['x'].flatten()
    y = data_transduceur['y'].flatten()
    z = data_transduceur['z'].flatten()
    lat, lon, elev = ecef2geodetic(x, y, z)
    lat = lat[condition_gnss]
    lon = lon[condition_gnss]
    elev = elev[condition_gnss]

    traj_reel = GNSS_trajectory(lat, lon, elev)
    slant_range, slant_range_cst, difference = calculate_travel_times_optimized(traj_reel, xyzR)
    data_DOG = sio.loadmat('../../data/DOG/DOG1-camp.mat')
    data_DOG = data_DOG["tags"].astype(float)
    acoustic_DOG = np.unwrap(data_DOG[:,1]/1e9*2*np.pi)/(2*np.pi)
    offset =  68200
    time_DOG = (data_DOG[:,0]+ offset)/3600
    condition_DOG = ((data_DOG[:,0] + offset)/3600 >= 25) & ((data_DOG[:,0] + offset)/3600 <= 38.5)
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
    axes[0, 1].set_xlim(25, 41)
    axes[0, 1].text(25, max(acoustic_DOG), label_text)  # This puts the label at x=25 and y=max(acoustic_DOG)
    axes[0, 1].legend()

    # Bottom plot (Difference)
    axes[1, 1].scatter(time_GNSS, difference_data, s=1, label='Difference (slant_range - matched_acoustic_DOG)')
    axes[1, 1].set_xlabel('Time [h]')
    axes[1, 1].set_xlim(25, 41)
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

    # 
    # initial_guess = [31.45, 291.32, 5225]
    # result = minimize(objective_function, initial_guess, args=(traj_reel, valid_acoustic_DOG, time_GNSS))
    #
    # optimal_xyzR = result.x
    # print("\n Optimal xyzR:", optimal_xyzR)
