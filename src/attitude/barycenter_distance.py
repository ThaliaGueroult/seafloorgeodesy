import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import math

def load_and_process_data(path):
    """Load and process data from a unit."""
    data = sio.loadmat(path)
    days = data['days'].flatten() - 59015
    times = data['times'].flatten()
    datetimes = ((days * 24 * 3600) + times) / 3600
    x, y, z = data['x'].flatten(), data['y'].flatten(), data['z'].flatten()
    return datetimes, x, y, z

paths = [
    '../../data/SwiftNav_Data/Unit1-camp_bis.mat',
    '../../data/SwiftNav_Data/Unit2-camp_bis.mat',
    '../../data/SwiftNav_Data/Unit3-camp_bis.mat',
    '../../data/SwiftNav_Data/Unit4-camp_bis.mat'
]

# Load data from all files
all_data = [load_and_process_data(path) for path in paths]

# Find common datetimes
common_datetimes = set(all_data[0][0])
for data in all_data[1:]:
    common_datetimes.intersection_update(data[0])
common_datetimes = sorted(common_datetimes)

# Apply mask to keep only common datetimes and corresponding values
filtered_data = []
for datetimes, x, y, z in all_data:
    mask = np.isin(datetimes, common_datetimes)
    filtered_data.append((datetimes[mask], x[mask], y[mask], z[mask]))

# Calculate distances between each pair of antennas
distances = {}
pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
for i, j in pairs:
    xi, yi, zi = filtered_data[i][1], filtered_data[i][2], filtered_data[i][3]
    xj, yj, zj = filtered_data[j][1], filtered_data[j][2], filtered_data[j][3]
    d = np.sqrt((xi - xj)**2 + (yi - yj)**2 + (zi - zj)**2)
    mean_distance = np.mean(d)
    distances[f"Antenna {i+1}-{j+1}"] = mean_distance

# Display average distances
for key, value in distances.items():
    print(f"Average distance between {key}: {value:.2f} m")

# Given distances
d14 = 10.20
d34 = 7.11
d24 = 11.35
d13 = 12.20
d12 = 4.93
d23 = 10.24

# Set position of Antenna 4
A4 = (0, 0)

# Antenna 1 is d14 meters from A4 along the positive x-axis
A1 = (d14, 0)

# Antenna 3 is d34 meters from A4 along the negative y-axis (from aft to starboard)
A3 = (0, -d34)

# For Antenna 2
y2 = -math.sqrt(d24**2 - d14**2)  # Using Pythagorean theorem, with a negative sign for the y-axis
A2 = (d14, y2)

print("Position of Antenna 1:", A1)
print("Position of Antenna 2:", A2)
print("Position of Antenna 3:", A3)
print("Position of Antenna 4 (Origin):", A4)

# Calculate the centroid
Bx = (A1[0] + A2[0] + A3[0] + A4[0]) / 4
By = (A1[1] + A2[1] + A3[1] + A4[1]) / 4
B = (Bx, By)

# New coordinates relative to the centroid
A1_new = (A1[0] - Bx, A1[1] - By)
A2_new = (A2[0] - Bx, A2[1] - By)
A3_new = (A3[0] - Bx, A3[1] - By)
A4_new = (A4[0] - Bx, A4[1] - By)

print("Centroid:", B)
print("Position of Antenna 1 relative to centroid:", A1_new)
print("Position of Antenna 2 relative to centroid:", A2_new)
print("Position of Antenna 3 relative to centroid:", A3_new)
print("Position of Antenna 4 relative to centroid:", A4_new)

# Initialize the plot
plt.figure(figsize=(10, 10))
plt.grid(True)
plt.axhline(y=0, color='k')  # Horizontal line
plt.axvline(x=0, color='k')  # Vertical line

# Plot antennas
plt.scatter(*A1_new, color='red', label=f"Antenna 1 {A1_new}")
plt.scatter(*A2_new, color='blue', label=f"Antenna 2 {A2_new}")
plt.scatter(*A3_new, color='green', label=f"Antenna 3 {A3_new}")
plt.scatter(*A4_new, color='black', label=f"Antenna 4 (Origin) {A4_new}")
plt.scatter(0, 0, color='purple', label="Center of Mass", s=100, marker='*')

# Titles and legends
plt.title("Antenna Arrangement on the Boat")
plt.xlabel("Forward <-> Aft")
plt.ylabel("Starboard <-> Port")
plt.legend()

# Show the plot
plt.show()
