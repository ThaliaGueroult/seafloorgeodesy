import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Load the data (simulated in this example)
data = np.loadtxt('../../dat/cast2.txt')
depth = data[:, 0]
sv = data[:, 1]

# Find the index where depth is maximal and use data until max_depth_index
max_depth_index = np.argmax(depth)
depth_descending = depth[:max_depth_index + 1]
sv_descending = sv[:max_depth_index + 1]

# Sample 100 evenly spaced points (or fewer if the array is smaller)
num_points = 100
indices = np.linspace(0, len(depth_descending) - 1, num_points, dtype=int)
depth_sampled = depth_descending[indices]
sv_sampled = sv_descending[indices]

# Plot sampled data
plt.figure()
plt.plot(sv_sampled, depth_sampled)
plt.xlabel('Sound Velocity (sv)')
plt.ylabel('Depth')
plt.title('Plot of sv and Depth')
plt.gca().invert_yaxis()  # Invert the y-axis if necessary
plt.show()

def preprocess_data(cz, depth):
    """Preprocess data by interpolating it using cubic interpolation."""
    z = np.linspace(0, 5250, int(1e6)+4784)
    f = interp1d(depth, cz, 'cubic', fill_value='extrapolate')
    cz = f(z)
    depth = z
    gradc = np.diff(cz) / np.diff(depth)
    gradc = np.insert(gradc, 0, gradc[0])
    return cz, depth, gradc

cz, depth, gradc = preprocess_data(sv_sampled, depth_sampled)

# Plot preprocessed data
plt.figure()
plt.plot(cz, depth)
plt.xlabel('Sound Velocity (sv)')
plt.ylabel('Depth')
plt.title('Plot of sv and Depth after Preprocessing')
plt.gca().invert_yaxis()  # Invert the y-axis if necessary
plt.show()

np.savetxt('../../data/cz_cast2_big.txt', cz)
np.savetxt('../../data/depth_cast2_big.txt', depth)
np.savetxt('../../data/gradc_cast2_big.txt', gradc)

def ray_analytic(cz, depth, gradc, theta0):
    """Compute the analytic ray tracing."""
    # Convert incident angle to radians
    theta0_rad = (90 - theta0) * np.pi / 180
    # Calculate ray parameter
    p = np.sin(theta0_rad) / cz[0]
    # Calculate horizontal positions and travel times along the ray path
    xk = np.cumsum(1 / (p * gradc[:-1]) * (np.sqrt(1 - p**2 * cz[:-1]**2) - np.sqrt(1 - p**2 * cz[1:]**2)))
    tk = np.cumsum(1 / gradc[:-1] * np.log((cz[1:] / cz[:-1]) * ((1 + np.sqrt(1 - p**2 * cz[:-1]**2)) / (1 + np.sqrt(1 - p**2 * cz[1:]**2)))))
    # Combine the source position with initial values
    x = np.concatenate(([0], xk))
    t = np.concatenate(([0], tk))
    return x, depth, t

# Plotting the ray tracing results for angles from 0 to 90 degrees with increments of 10
plt.figure()
for angle in range(0, 91, 10):
    x, z, t = ray_analytic(cz, depth, gradc, angle)
    plt.plot(x, z, label=f'Angle = {angle}°')
plt.gca().invert_yaxis()  # Invert the y-axis
plt.xlabel('X-Coordinate')
plt.ylabel('Depth')
plt.legend()
plt.show()
