#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy import integrate
import math

#inspired from August Wietfield, 2022

'''
uses an observed piecewise-linear (in depth) wavespeed profile to calculate and plot
the paths of soundwave rays

plots rays for a given set of initial conditions and given minimum and maximum
depths, horizontal distances
'''

depth = np.loadtxt('../../data/depth_GDEM.txt')
soundspeedData = np.loadtxt('../../data/sv_GDEM.txt')
m = np.diff(soundspeedData) / np.diff(depth)
m = np.insert(m, 0, m[0])

# define wavespeed profile function
def c(z):
    segment = np.searchsorted(depth, z) # - 1
    if segment >= depth.shape[0] : segment = depth.shape[0] - 1
    return m[segment] * (z - depth[segment]) + soundspeedData[segment]

# define derivative of wavespeed profile function
def dcdz(z):
    segment = np.searchsorted(depth, z) # - 1
    if segment >= depth.shape[0] : segment = depth.shape[0] - 1
    return m[segment]

def ydot(y, t):
    i, x, z, p = y
    idot = p * c(z) * dcdz(z)
    xdot = c(z) * math.sin(i)
    zdot = c(z) * math.cos(i)
    pdot = 0

    return [idot, xdot, zdot, pdot]

def rayPathLSODA(alpha, z0=0, x0=10, tMax=400, dt=0.01, xr=12000, zr=5100, tol=1e-6):
    # process inputs
    i0 = (90 - alpha) * np.pi / 180
    p = math.sin(i0) / c(z0)

    # initial state vector
    y0 = [i0, x0, z0, p]

    # independent variable list (arclength)
    tVal = 0
    tVals = []
    while tVal < tMax:
        tVals.append(tVal)
        tVal += dt

    # calculation of ray by integrating ydot (uses LSODA algorithm)
    ray = integrate.odeint(ydot, y0, tVals)

    # slice ray for x-values, z-values
    x = ray[:, 1]
    z = ray[:, 2]

    # clean rays for surfacing, i.e., z < 0 (depth is positive)
    surfaced = False
    for k in range(z.size):
        if abs(x[k] - xr) < tol and abs(z[k] - zr) < tol:
            break  # Exit the loop if the condition is satisfied
        if surfaced:
            z[k] = np.nan
            x[k] = np.nan
            continue
        if z[k] < 0:
            surfaced = True
            z[k] = np.nan
            x[k] = np.nan
        # Check if (x[k], z[k]) is close enough to (xr, zr)
    return x, z

def rayPlot(alpha, zInit="default", xInit="default", tMax=400, dt=0.01):
    # if zInit and xInit are "default", zInit and xInit set as 0
    if isinstance(zInit, str):
        zInit = np.zeros(alpha.size)
    if isinstance(xInit, str):
        xInit = np.zeros(alpha.size)
    print(alpha)
    xInits = xInit
    zInits = zInit

    # Initialisation du graphique
    fig, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [2, 1]})

    # Graphique de gauche, trajectoires des rayons
    for k, alpha_k in enumerate(alpha):
        print(alpha_k)
        xPath, zPath = rayPathLSODA(alpha_k, zInits[k], xInits[k], tMax, dt)
        ax1.plot(xPath, zPath, label='{}°'.format(alpha_k))

    # Graphique de droite, profil de vitesse du son
    zVals = np.linspace(minDepth, maxDepth, 1000)
    cVals = [c(zVal) for zVal in zVals]
    ax2.plot(cVals, zVals)

    # Configuration des fenêtres des graphiques
    ax1.set_xlim(0, maxX)
    ax1.set_ylim(minDepth, maxDepth)
    ax1.invert_yaxis()
    ax1.legend()
    ax2.set_ylim(minDepth, maxDepth)
    ax2.invert_yaxis()

    # Titres et étiquettes des axes
    ax1.set_title("Ray Path in Linear Wavespeed profile")
    ax1.set_xlabel("Horizontal Distance (km)")
    ax1.set_ylabel("Depth (km)")

    ax2.set_title("Sound Celerity Profile")
    ax2.set_xlabel("Sound Celerity (km/s)")
    ax2.set_ylabel("Depth (km)")

    # Ajustement de la taille de la figure
    fig.set_size_inches(12, 4)
    fig.set_dpi(100)

    # Affichage du graphique
    plt.show()

if __name__=='__main__':
    # initial conditions for plotting rays
    alpha = np.linspace(10, 170) # degrees below Horizontal

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
    soundspeedData = np.loadtxt('../../data/sv_GDEM.txt')

    # define depths
    depth = np.loadtxt('../../data/depth_GDEM.txt')

    # get slopes of segments of linear model
    m = np.diff(soundspeedData) / np.diff(depth)
    m = np.insert(m, 0, m[0])

    alpha = np.linspace(0, 90, num=10)
    rayPlot(alpha)
