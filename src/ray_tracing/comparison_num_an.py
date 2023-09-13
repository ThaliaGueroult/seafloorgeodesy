import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy import integrate
import math
import time
from scipy.interpolate import interp1d

# Load depth and sound speed data
depth = np.loadtxt('../../data/SV/depth_cast2_big.txt')
soundspeedData = np.loadtxt('../../data/SV/cz_cast2_big.txt')

# Calculate the slope of the wavespeed profile
m = np.diff(soundspeedData) / np.diff(depth)
m = np.insert(m, 0, m[0])


# Define the wavespeed profile function
def c(z):
    # Find the segment of the wavespeed profile based on depth
    segment = np.searchsorted(depth, z)
    if segment >= depth.shape[0]:
        segment = depth.shape[0] - 1
    # Calculate the wavespeed using the linear interpolation equation
    return m[segment] * (z - depth[segment]) + soundspeedData[segment]

# Define the derivative of the wavespeed profile function
def dcdz(z):
    # Find the segment of the wavespeed profile based on depth
    segment = np.searchsorted(depth, z)
    if segment >= depth.shape[0]:
        segment = depth.shape[0] - 1
    # Return the slope of the segment
    return m[segment]

# Define the derivative function for integrating the ray equations
def ydot(y, t):
    i, x, z, p = y
    idot = p * c(z) * dcdz(z)
    xdot = c(z) * math.sin(i)
    zdot = c(z) * math.cos(i)
    pdot = 0
    return [idot, xdot, zdot, pdot]

# Calculate the ray paths using numerical integration
def ray_numeric(alpha, z0=0, x0=0, tMax=40, dt=0.001):
    # Convert the angle to radians and calculate the initial slope
    i0 = (90 - alpha) * np.pi / 180
    p = math.sin(i0) / c(z0)
    # Set the initial state vector
    y0 = [i0, x0, z0, p]
    # Generate the time values for integration
    tVal = 0
    tVals = []
    while tVal < tMax:
        tVals.append(tVal)
        tVal += dt
    # Integrate the ray equations using LSODA algorithm
    ray = integrate.odeint(ydot, y0, tVals)
    # Extract the x and z coordinates of the ray
    x = ray[:, 1]
    z = ray[:, 2]
    # Remove the portions of the ray that surface (z < 0)
    surfaced = False
    for k in range(z.size):
        if surfaced:
            z[k] = np.nan
            x[k] = np.nan
            continue
        if z[k] < 0:
            surfaced = True
            z[k] = np.nan
            x[k] = np.nan
    return (x, z, tVals)


def ray_analytic(cz, depth, gradc, theta0):
    # Convert incident angle to radians and calculate the ray parameter
    theta0_rad = np.radians(90 - theta0)
    p = np.sin(theta0_rad) / cz[0]
    p_squared = p ** 2

    # Pre-calculate commonly used values
    cz_1 = cz[:-1]
    cz_2 = cz[1:]
    gradc_sub = gradc[:-1]

    sqrt_term_1 = np.sqrt(1 - p_squared * cz_1 ** 2)
    sqrt_term_2 = np.sqrt(1 - p_squared * cz_2 ** 2)

    # Calculate the horizontal positions along the ray path
    xk = np.cumsum(1 / (p * gradc_sub) * (sqrt_term_1 - sqrt_term_2))

    # Calculate the travel times along the ray path
    log_term = np.log((cz_2 / cz_1) * ((1 + sqrt_term_1) / (1 + sqrt_term_2)))
    tk = np.cumsum(1 / gradc_sub * log_term)

    # Combine the source position with the initial values
    x = np.concatenate(([0], xk))
    t = np.concatenate(([0], tk))

    # Return the x-coordinates, depth values, and travel times
    return x, depth, t

if __name__=='__main__':
    cz = np.loadtxt('../../data/SV/cz_cast2_big.txt')
    depth = np.loadtxt('../../data/SV/depth_cast2_big.txt')
    gradc = np.loadtxt('../../data/SV/gradc_cast2_big.txt')
    # Calculate the ray paths for different angles and time steps
    # Calculate the ray paths for different angles and time steps
    angles = [20, 45, 80]
    tMax = 15
    dt = 0.001
    tVals = np.arange(0, tMax, dt)

    plt.figure(figsize=(14, 6))

    for angle in angles:
        x_numeric, z_numeric, _ = ray_numeric(angle)
        x_analytic, z_analytic, _ = ray_analytic(cz, depth, gradc, angle)

        # Interpolation of the solution analytically in terms of depth (z)
        interpolator = interp1d(z_analytic, x_analytic, bounds_error=False, fill_value=np.nan)
        x_analytic_interp = interpolator(z_numeric)

        # Calculate the difference between numeric and interpolated analytic rays
        diff = x_numeric - x_analytic_interp

        # Plotting the ray comparisons
        plt.subplot(1, 2, 1)
        plt.plot(x_numeric[:5000], z_numeric[:5000], label=f'Numeric - Angle {angle}°', linestyle='-', linewidth=5, alpha=0.5)
        plt.plot(x_analytic, z_analytic, label=f'Analytic - Angle {angle}°', linestyle='--', alpha=1)
        plt.xlabel('Horizontal Distance x (m)')
        plt.ylim(0,5250)
        plt.ylabel('Depth (m)')
        plt.title('Ray Comparison: Numeric vs Analytic')
        plt.gca().invert_yaxis()
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(diff, z_numeric, label=f'Angle {angle}°')
        plt.xlabel('Ray Difference (m)')
        plt.ylabel('Depth (m)')
        plt.ylim(0,5250)
        plt.title('Difference between Numeric and Analytic Rays')
        plt.gca().invert_yaxis()
        plt.legend()
    plt.tight_layout()
    plt.show()
