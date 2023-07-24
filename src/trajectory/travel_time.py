import matplotlib.pyplot as plt
import numpy as np
import math

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

def calculate_travel_times(trajectory, receiver_lat, receiver_lon, receiver_elev, speed_of_sound):
    '''
    Calculate the travel times from each point in the trajectory to the receiver point.

    Args:
    trajectory (list): List of tuples representing the trajectory points in the format (latitude, longitude, altitude).
    receiver_lat (float): Latitude of the receiver point.
    receiver_lon (float): Longitude of the receiver point.
    receiver_elev (float): Elevation of the receiver point.
    speed_of_sound (float): Speed of sound in m/s.

    Returns:
    travel_times (list): List of travel times in seconds for each point in the trajectory.
    '''

    travel_times = []

    for point in trajectory:
        lat, lon, elev = point
        elev = 0
        # Calculate the Euclidean distance between the points
        distance = math.sqrt((receiver_lat - lat)**2 + (receiver_lon - lon)**2 + (receiver_elev - elev)**2)

        # Calculate the travel time based on the distance and speed of sound
        travel_time = distance / speed_of_sound
        travel_times.append(travel_time)

    return travel_times

# Generate the trajectory
trajectory = generate_trajectory()

# Receiver point coordinates and elevation
receiver_lat = 31.45
receiver_lon = 291.13
receiver_elev = 5225

# Speed of sound in m/s
speed_of_sound = 1515

# Calculate travel times
travel_times = calculate_travel_times(trajectory, receiver_lat, receiver_lon, receiver_elev, speed_of_sound)

# Plot the successive travel times
plt.plot(travel_times)
plt.xlabel('Point Index')
plt.ylabel('Travel Time (s)')
plt.title('Successive Travel Times to Receiver Point')
plt.grid(True)
plt.show()
