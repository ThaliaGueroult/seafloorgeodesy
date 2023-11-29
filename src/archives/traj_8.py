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
