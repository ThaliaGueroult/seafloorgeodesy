import numpy as np
import matplotlib.pyplot as plt
import scipy
from geodesy_toolbox import *

def calculate_intermediate_point(xyzS, xyzR):
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
    xyzI = calculate_intermediate_point(xyzS, xyzR)
    xs, ys, zs = xyzS[0], xyzS[1], xyzS[2]
    xi, yi, zi = xyzI[0], xyzI[1], xyzI[2]

    # Calcul de la distance verticale
    dz = zi - zs

    # Calcul de la distance horizontale
    dx = xi - xs
    dy = yi - ys
    distance_horizontale = np.sqrt(dx**2 + dy**2)

    # Calcul de l'angle d'élévation
    beta = math.degrees(math.atan2(dz, distance_horizontale))

    return dz, beta

if __name__ == '__main__':
