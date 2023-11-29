#!/usr/bin/python3
import os
import numpy as np
import scipy.io as sio
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import pyproj

# Load the Effective Sound Velocity (ESV) matrix from a .mat file
folder_path = '../esv/castbermuda/global_table_interp'
matrix_file_path = os.path.join(folder_path, 'global_table_esv.mat')
data = sio.loadmat(matrix_file_path)

dz_array = data['distance'].flatten()
angle_array = data['angle'].flatten()
esv_matrix = data['matrice']

def latlon_to_utm(lat, lon):
    wgs84 = pyproj.CRS("EPSG:4326")
    utm_zone_19R = pyproj.CRS("EPSG:32619")
    transformer = pyproj.Transformer.from_crs(wgs84, utm_zone_19R, always_xy=True)
    easting, northing = transformer.transform(lon, lat)
    return easting, northing

def calculate_distance_xyz(xyzS, xyzR):
    return np.linalg.norm(xyzS - xyzR, axis=1)

def find_esv(dz_array, angle_array, esv_matrix, beta, dz):
    idx_closest_dz = np.argmin(np.abs(dz_array[:, None] - dz), axis=0)
    idx_closest_beta = np.argmin(np.abs(angle_array[:, None] - beta), axis=0)
    closest_esv = esv_matrix[idx_closest_dz, idx_closest_beta]
    return closest_esv

def calculate_travel_times_optimized(trajectory, xyzR):
    distances = calculate_distance_xyz(trajectory, xyzR)
    dz =  xyzR[2] - trajectory[:, 2]
    dh = np.linalg.norm(trajectory[:, :2] - xyzR[:2], axis=1)
    beta = np.degrees(np.arctan2(dz, dh))
    esv = find_esv(dz_array, angle_array, esv_matrix, beta, dz)
    travel_times = distances / esv
    travel_times_cst = distances / 1515
    diff = travel_times_cst - travel_times
    return travel_times, travel_times_cst, diff

def objective_function(xyzR, traj_reel, valid_acoustic_DOG, time_GNSS):
    slant_range, _, _ = calculate_travel_times_optimized(traj_reel, xyzR)
    difference_data = slant_range - valid_acoustic_DOG
    result = np.sqrt(np.nanmean(difference_data**2))
    return result

if __name__ == '__main__':
    # Convert xyzR to UTM
    xyzR = [  31.46358708,  291.29857339, 5209.190554  ]
    # Assuming you have lat, lon, elev in xyzR
    easting, northing = latlon_to_utm(xyzR[0], xyzR[1])
    xyzR_utm = [easting, northing, xyzR[2]]
    print(xyzR_utm)
    data_transduceur = sio.loadmat('transducer_positions.mat')
    print(data_transduceur)
    datetimes = data_transduceur['times'].flatten()
    condition_gnss = (datetimes >= 25) & (datetimes <= 38)
    time_GNSS = datetimes[condition_gnss]

    x, y = latlon_to_utm(data_transduceur['lat'].flatten()[condition_gnss], data_transduceur['lon'].flatten()[condition_gnss])
    z = data_transduceur['elev'].flatten()[condition_gnss]
    traj_reel = np.column_stack((x, y, z))

    slant_range, slant_range_cst, difference = calculate_travel_times_optimized(traj_reel, xyzR_utm)

    data_DOG = sio.loadmat('../../data/DOG/DOG1-camp.mat')
    data_DOG = data_DOG["tags"].astype(float)
    acoustic_DOG = np.unwrap(data_DOG[:,1]/1e9*2*np.pi)/(2*np.pi)
    offset = 68126
    time_DOG = (data_DOG[:,0]+ offset)/3600
    condition_DOG = ((data_DOG[:,0] + offset)/3600 >= 25) & ((data_DOG[:,0] + offset)/3600 <= 40.9)
    time_DOG = time_DOG[condition_DOG]
    acoustic_DOG = acoustic_DOG[condition_DOG]

    valid_acoustic_DOG = np.full(time_GNSS.shape, np.nan)

    common_indices = np.isin(time_GNSS, time_DOG)
    indices_in_DOG = np.searchsorted(time_DOG, time_GNSS[common_indices])
    valid_acoustic_DOG[common_indices] = acoustic_DOG[indices_in_DOG]

    difference_data = slant_range - valid_acoustic_DOG
    RMS = np.sqrt(np.nanmean(difference_data**2))
    print('\n RMS{} s'.format(RMS))

    # Visualization code
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), gridspec_kw={'width_ratios': [1, 4], 'height_ratios': [4, 1]}, sharey='row')
    label_text = f"Lat: {xyzR[1]}, Lon: {xyzR[0]}, elev: {xyzR[2]}"
    axes[0, 1].scatter(time_DOG, acoustic_DOG, s=3, label='Acoustic DOG')
    axes[0, 1].scatter(time_GNSS, slant_range, s=1, label='GNSS SV Bermuda')
    axes[0, 1].set_ylabel('Time Travel (s)')
    axes[0, 1].set_xlabel('Time [h]')
    axes[0, 1].set_xlim(25, 41)
    axes[0, 1].text(25, max(acoustic_DOG), label_text)
    axes[0, 1].legend()

    axes[1, 1].scatter(time_GNSS, difference_data, s=1, label='Difference (slant_range - matched_acoustic_DOG)')
    axes[1, 1].set_xlabel('Time [h]')
    axes[1, 1].set_xlim(25, 41)
    axes[1, 1].text(25,10,'RMS = {} s'.format(RMS))
    axes[1, 1].legend()

    axes[1, 0].hist(difference_data, orientation='horizontal', bins=30, alpha=0.5)
    axes[1, 0].set_xlabel('Frequency')
    axes[1, 0].set_title('Histogram')
    axes[1, 0].invert_xaxis()

    axes[1, 0].get_shared_y_axes().join(axes[1, 0], axes[1, 1])
    axes[0, 0].axis('off')

    plt.tight_layout()
    plt.show()

    # Minimization (if needed)
    initial_guess = [easting, northing, xyzR[2]]
    result = minimize(objective_function, initial_guess, args=(traj_reel, valid_acoustic_DOG, time_GNSS))
    print(result)
    optimal_xyzR = result.x
    print("\n Optimal xyzR:", optimal_xyzR)
