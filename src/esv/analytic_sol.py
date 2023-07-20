import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.interpolate import interp1d
from numba import jit
from scipy.interpolate import CubicSpline

sv=np.loadtxt('../../data/sv_GDEM.txt')
depth=np.loadtxt('../../data/depth_GDEM.txt')

np.seterr(invalid='ignore')

def ray_analytic(cz, depth, theta0):
    '''
    Inputs:
        cz: Sound speed profile
        depth: Depth values
        theta0: Incident angle in degrees
    Outputs:
        x: x-coordinates of the ray path
        z: Depth values
        t: Travel time along the ray path
    '''
    # Generate depth values
    z = np.linspace(0, np.max(depth) + 25.0, int(1e6))

    # Interpolate sound speed profile
    f = interp1d(depth, cz, 'linear', bounds_error=False, fill_value='extrapolate')
    cz = f(z)
    depth = z

    # Calculate the gradient of the sound speed profile
    gradc = np.diff(cz) / np.diff(depth)
    gradc = np.insert(gradc, 0, gradc[0])

    # Convert incident angle to radians
    theta0 = (90 - theta0) * np.pi / 180
    # Calculate the ray parameter
    p = np.sin(theta0) / cz[0]

    # Calculate the horizontal positions along the ray path
    xk = np.cumsum(1 / (p * gradc[:-1]) * (np.sqrt(1 - p**2 * cz[:-1]**2) - np.sqrt(1 - p**2 * cz[1:]**2)))
    # Calculate the travel times along the ray path
    tk = np.cumsum(1 / gradc[:-1] * np.log((cz[1:] / cz[:-1]) * ((1 + np.sqrt(1 - p**2 * cz[:-1]**2)) / (1 + np.sqrt(1 - p**2 * cz[1:]**2)))))

    # Combine the source position with the initial values
    x = np.concatenate(([0], xk))
    t = np.concatenate(([0], tk))

    # Return the x-coordinates, depth values, and travel times
    return x, z, t


if __name__=='__main__':
    sv=np.loadtxt('../../data/sv_GDEM.txt')
    depth=np.loadtxt('../../data/depth_GDEM.txt')
    x, z, t=ray_analytic(sv,depth,10)
