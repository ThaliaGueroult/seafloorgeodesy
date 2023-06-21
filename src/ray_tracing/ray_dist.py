import numpy as np
import scipy.io as sio
import math
from analytic_sol import *
from scipy.optimize import minimize
from tqdm import tqdm

def dist(xyzS, xyzR):
    '''
    Inputs :
        xyzS : position of the source
        xyzR : position of the receiver
    Outputs :
        dist : distance between source and receiver
        time : travel time
    '''
    xs, ys, zs = xyzS[0], xyzS[1], xyzS[2]
    xr, yr, zr = xyzR[0], xyzR[1], xyzR[2]
    alpha = np.arange(9.95, 10.00, 0.001)
    progress_bar = tqdm(total=len(alpha), desc="Progress", unit="alpha")
    for i in range(len(alpha)):
        x, z = rayPathLSODA(alpha[i], zs, xs, tMax=40, dt=0.01)
        t = np.arange(0,400,0.01)
        errors = np.sqrt((xr - x) ** 2 + (zr - z) ** 2)
        id_min = np.nanargmin(errors)
        err_min = np.nanmin(errors)
        if err_min < 2 :
            dist = np.sqrt((xr-x[id_min])**2 + (zr - z[id_min])**2)
            plt.scatter(xs, zs, color='green', label='Source')
            plt.scatter(xr, zr, color='red', label='Receiver')
            plt.plot(x, z, label='Alpha = {}°'.format(alpha[i]))
            plt.gca().invert_yaxis()
            plt.title("Ray Path in Linear Wavespeed profile")
            plt.xlabel("Horizontal Distance (km)")
            plt.ylabel("Depth (km)")
            plt.legend()
            plt.show()
            return (print('xcalc={} and zcalc={} at t={} for alpha={} and dist ={}'.format(x[id_min], z[id_min], t[id_min], alpha[i], dist)))
        progress_bar.update(1)
    return (print('No ray reaches the receiver, change angle or tolerance threshold'))

if __name__ == '__main__':
    dist([0, 0, 0], [22761, 3000, 5025])
