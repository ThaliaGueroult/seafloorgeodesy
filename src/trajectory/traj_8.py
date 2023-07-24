# import math
# import matplotlib.pyplot as plt
#
# def generate_trajectory():
#     '''
#     Generate a trajectory in the shape of an "8" with varying altitude.
#
#     Returns:
#     trajectory (list): List of tuples representing the trajectory points in the format (latitude, longitude, altitude).
#     '''
#
#     lat_min = 31.35
#     lat_max = 31.55
#     lon_min = 291.20
#     lon_max = 291.40
#
#     lat_range = lat_max - lat_min
#     lon_range = lon_max - lon_min
#
#     num_points = 10000  # Number of points on the trajectory
#     angle_factor = 20  # Scale factor for the angle
#
#     trajectory = []
#
#     for i in range(num_points):
#         t = float(i) / (num_points - 1)  # Normalized value between 0 and 1
#         angle = t * 2 * math.pi * angle_factor
#         x = math.sin(angle)
#         y = math.sin(angle * 2) / 2
#
#         lat = lat_min + (x + 1) * (lat_range / 2)
#         lon = lon_min + (y + 1) * (lon_range / 2)
#         elev = 5 * math.sin(angle)  # Sinusoidal altitude between -5 m and 5 m
#
#         point = (lat, lon, elev)
#         trajectory.append(point)
#
#     return trajectory
#
# # Generate the trajectory
# trajectory = generate_trajectory()
#
# # Extract latitude, longitude, and altitude coordinates for plotting
# lats = [point[0] for point in trajectory]
# lons = [point[1] for point in trajectory]
# elevs = [point[2] for point in trajectory]
#
# # Plot the trajectory with a colormap for altitude
# plt.scatter(lons, lats, c=elevs, cmap='jet', s=5)
# plt.xlabel('Longitude')
# plt.ylabel('Latitude')
# plt.title('Trajectory in the shape of an "8" with colormap for altitude (shorter period)')
# plt.colorbar(label='Altitude')
# plt.grid(True)
# plt.show()


import math
import matplotlib.pyplot as plt

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

# Generate the trajectory
trajectory = generate_trajectory()

# Extract latitude, longitude, and altitude coordinates for plotting
lats = [point[0] for point in trajectory]
lons = [point[1] for point in trajectory]
elevs = [point[2] for point in trajectory]

# Plot the trajectory with a colormap for altitude
plt.scatter(lons, lats, c=elevs, cmap='jet', s=5)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Trajectory in the shape of an "8" with colormap for altitude')
plt.colorbar(label='Altitude')
plt.grid(True)
plt.show()
