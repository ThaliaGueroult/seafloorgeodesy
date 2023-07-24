import numpy as np
import matplotlib.pyplot as plt
import scipy
from geodesy_toolbox import *

# This script calculates the intermediate point and vertical distance
# between a source (xyzS) and a receiver (xyzR) using geodetic coordinates.

def calculate_intermediate_point(xyzS, xyzR):
    '''
    Inputs:
    xyzS - Source coordinates [longitude, latitude, altitude]
    xyzR - Receiver coordinates [longitude, latitude, altitude]

    Outputs:
    xyzI - Intermediate point coordinates [longitude, latitude, altitude]
    '''
    # Convert source and receiver coordinates from geodetic to ECEF
    xs, ys, zs = geod2ecef(xyzS[0], xyzS[1], xyzS[2])
    xr, yr, zr = geod2ecef(xyzR[0], xyzR[1], xyzR[2])

    # Calculate the distance between the source and receiver
    d = np.sqrt((xr - xs) ** 2 + (yr - ys) ** 2)

    # Calculate the coordinates of the intermediate point I
    xi = xs + d
    yi = ys
    zi = zr

    xyzI = [xi, yi, zr]

    return xyzI

def xyz2dec(xyzS, xyzR):
    '''
    Inputs:
    xyzS - Source coordinates [longitude, latitude, altitude]
    xyzR - Receiver coordinates [longitude, latitude, altitude]

    Outputs:
    dz - Vertical distance between source and receiver
    beta - Elevation angle from source to receiver
    '''
    # Calculate the intermediate point coordinates
    xyzI = calculate_intermediate_point(xyzS, xyzR)
    xs, ys, zs = xyzS[0], xyzS[1], xyzS[2]
    xi, yi, zi = xyzI[0], xyzI[1], xyzI[2]

    # Calculate the vertical distance
    dz = zi - zs

    # Calculate the horizontal distance
    dx = xi - xs
    dy = yi - ys
    distance_horizontale = np.sqrt(dx**2 + dy**2)

    # Calculate the elevation angle
    beta = math.degrees(math.atan2(dz, distance_horizontale))

    return dz, beta

if __name__ == '__main__':
