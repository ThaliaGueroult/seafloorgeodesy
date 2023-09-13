#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy import integrate
import math
import time
# Inspired from August Wietfield, 2022

'''
This code calculates and plots soundwave ray paths using a piecewise-linear wavespeed profile.
'''

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

# Plot the ray paths
def rayPlot(alpha, zInit="default", xInit="default", tMax=100, dt=0.01):
    t=time.time()
    # Set the initial positions of the rays
    if isinstance(zInit, str):
        zInit = np.zeros(alpha.size)
    if isinstance(xInit, str):
        xInit = np.zeros(alpha.size)
    xInits = xInit
    zInits = zInit
    # Create subplots for ray paths and wavespeed profile
    fig, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [2, 1]})
    # Plot the ray paths
    for k, alpha_k in enumerate(alpha):
        xPath, zPath, tPath = ray_numeric(alpha_k, zInits[k], xInits[k], tMax, dt)
        ax1.plot(xPath, zPath, label='{}°'.format(alpha_k))
    # Plot the wavespeed profile
    zVals = np.linspace(minDepth, maxDepth, int(1e6))
    cVals = [c(zVal) for zVal in zVals]
    ax2.plot(cVals, zVals)
    # Configure the plot settings
    # ax1.scatter(22761, 5025, color='red', label='Receiver')
    ax1.set_xlim(0, 32000)
    ax1.set_ylim(minDepth, maxDepth)
    ax1.invert_yaxis()
    ax1.legend()
    ax2.set_ylim(minDepth, maxDepth)
    ax2.invert_yaxis()
    ax1.set_title("Ray Path numeric in Linear Wavespeed profile")
    ax1.set_xlabel("Horizontal Distance (km)")
    ax1.set_ylabel("Depth (km)")
    ax2.set_title("Sound Celerity Profile")
    ax2.set_xlabel("Sound Celerity (km/s)")
    ax2.set_ylabel("Depth (km)")
    fig.set_size_inches(12, 4)
    fig.set_dpi(100)
    print(time.time()-t)
    plt.show()

if __name__=='__main__':
    # initial conditions for plotting rays
    alpha = np.linspace(0, 91, 10) # degrees below Horizontal

    # plotting windows

    # minimum and maximum depths (in m)
    minDepth = 0
    maxDepth = 6000

    # minimum and maximum horizontal depths (in m)
    minX = -100000
    maxX = 100000

    # upload soundspeed data
    # assuming you have the soundspeed profile in a numpy array called "soundspeedData"
    # with dimensions (numDepths,)
    soundspeedData = np.loadtxt('../../data/SV/cz_cast2_big.txt')

    # define depths
    depth = np.loadtxt('../../data/SV/depth_cast2_big.txt')

    # get slopes of segments of linear model
    m = np.diff(soundspeedData) / np.diff(depth)
    m = np.insert(m, 0, m[0])

    alpha = np.linspace(0, 90, num=10)
    rayPlot(alpha)
