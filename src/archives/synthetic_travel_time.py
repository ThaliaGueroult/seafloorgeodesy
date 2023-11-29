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
import matplotlib.animation as animation

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

def generate_lissajous_trajectory():
    """
    This function generates a Lissajous trajectory for the source.
    Output:
        Returns a DataFrame containing the trajectory
    """
    # Define parameters for the Lissajous trajectory
    lat_center = 31.45
    lon_center = 291.30
    elev_center = -37.5
    A = 0.1  # amplitude for x (Longitude)
    B = 0.2 # amplitude for y (Latitude)
    C = 1  # amplitude for z (Elevation)
    a = 3    # frequency for x, change the number of loop
    b = 5    # frequency for y, change the number of loop
    c = 1    # frequency for z
    delta = np.pi / 2  # phase shift

    # Generate the trajectory
    time = np.linspace(0, np.pi, 361)  # One full cycle
    lon_traj = lon_center + A * np.sin(a * time + delta)
    lat_traj = lat_center + B * np.cos(b * time)
    elev_traj = elev_center + C * np.sin(c * time)

    trajectory =  list(zip(lat_traj, lon_traj, elev_traj))

    return trajectory,time

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

def animate(i, sc, line, trajectory, time, travel_time, xyzR):
    lat = [point[0] for point in trajectory[:i]]
    lon = [point[1] for point in trajectory[:i]]
    elev = [point[2] for point in trajectory[:i]]

    # Update the trajectory scatter plot
    sc.set_offsets(np.c_[lon, lat])
    sc.set_array(np.array(elev))

    # Update the time vs travel_time line plot
    line.set_data(time[:i], travel_time[:i])

    return sc, line,

if __name__ == '__main__':
    trajectory, time = generate_lissajous_trajectory()
    xyzR = [31.46356396, 291.2985875, 5190.77000034]
    travel_time, _, _ = calculate_travel_times_optimized(trajectory, xyzR)
    # Extract the values for initial plotting
    lat = [point[0] for point in trajectory]
    lon = [point[1] for point in trajectory]
    elev = [point[2] for point in trajectory]

    fig, axs = plt.subplots(1, 2, figsize=(20, 8))

    # First subplot for trajectory
    sc = axs[0].scatter([], [], c=[], cmap='viridis', vmin=min(elev), vmax=max(elev))
    axs[0].scatter(xyzR[1], xyzR[0], color='r', label='beacon')
    axs[0].set_xlim(min(lon), max(lon))
    axs[0].set_ylim(min(lat), max(lat))
    axs[0].set_xlabel('Longitude')
    axs[0].set_ylabel('Latitude')
    axs[0].set_title('Lissajous Trajectory of the Source')
    axs[0].grid(True)
    plt.colorbar(sc, ax=axs[0], label='Elevation (m)')

    # Second subplot for time vs travel_time
    line, = axs[1].plot([], [], lw=2)
    axs[1].set_xlim(0, max(time))
    axs[1].set_ylim(0, max(travel_time))
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Travel Time')
    axs[1].set_title('Time vs Travel Time')
    axs[1].grid(True)

    ani = animation.FuncAnimation(fig, animate, frames=len(trajectory), fargs=(sc, line, trajectory, time, travel_time, xyzR),
                                  interval=100, blit=True)

    plt.tight_layout()
    plt.show()

    # Create the plot grid
    plt.figure(figsize=(20, 8))

    # First subplot for trajectory
    plt.subplot(1, 2, 1)
    sc = plt.scatter(lon, lat, c=elev, cmap='viridis')
    plt.scatter(xyzR[1], xyzR[0], color='r', label='beacon')
    plt.colorbar(sc, label='Elevation (m)')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Lissajous Trajectory of the Source')
    plt.grid(True)

    # Second subplot for time vs travel_time
    plt.subplot(1, 2, 2)
    plt.plot(time, travel_time)
    plt.xlabel('Time')
    plt.ylabel('Travel Time')
    plt.title('Time vs Travel Time')
    plt.grid(True)

    plt.tight_layout()  # Adjust layout so plots do not overlap
    plt.show()
