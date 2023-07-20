import numpy as np
import scipy.io as sio
from scipy.io import savemat
import math
from scipy.optimize import minimize
from tqdm import tqdm
from analytic_sol import *
import multiprocessing as mp

def dist_analytic(dz, beta):
    '''
    Inputs :
        dz : vertical distance between source and receiver
        beta : elevation angle between source and receiver
    Outputs :
        time : travel time
        c_esv : effective sound velocity between source and receiver
    '''

    # Compute horizontal distance between source and receiver
    beta = beta*np.pi/180
    dh = dz / np.tan(beta)
    xr, zr = dh, dz

    # Set initial parameters
    alpha_range = np.linspace(0, 90, 100)
    alpha_step = (max(alpha_range) - min(alpha_range)) / len(alpha_range)
    tolerance = 0.001

    # Initialize variables for tracking minimum distance and associated parameters
    dist_min = np.inf
    x_min, z_min = None, None
    t_receiver = None
    best_alpha = None
    x_fin = np.empty(0)
    z_fin = np.empty(0)

    # Set flags for plotting and early termination
    plot_results = False
    early_termination = False

    while alpha_step > 10e-10:
        alpha = np.array(alpha_range)
        progress_bar = tqdm(total=len(alpha), desc="Progress", unit="alpha")

        errors_all = np.empty((len(alpha),))
        x_all = np.empty((len(alpha),))
        z_all = np.empty((len(alpha),))
        t_all = np.empty((len(alpha),))

        for i, angle in enumerate(alpha):
            # Calculate ray path for the given angle
            x, z, t = ray_analytic(sv, depth, angle)

            # Calculate errors and find minimum error index
            errors = np.sqrt((xr - x) ** 2 + (zr - z) ** 2)
            errors_all[i] = np.nanmin(errors)
            id_min = np.nanargmin(errors)

            # Store corresponding coordinates and time for minimum error
            x_all[i] = x[id_min]
            z_all[i] = z[id_min]
            t_all[i] = t[id_min]

            # Calculate distance and update minimum distance if necessary
            dist = np.sqrt((xr - x_all[i]) ** 2 + (zr - z_all[i]) ** 2)
            if dist < dist_min:
                dist_min = dist
                x_min, z_min = x_all[i], z_all[i]
                best_alpha = angle
                t_receiver = t_all[i]

            progress_bar.update(1)

            # Check if tolerance condition is met for early termination
            if errors_all[i] < tolerance and x_min is not None and z_min is not None:
                x_fin, z_fin, t_fin = x, z, t
                early_termination = True
                break

        progress_bar.close()

        if early_termination :
            d_sr = dz / np.sin(beta)
            c_esv = d_sr / t_receiver
            print('xcalc={} and zcalc={} for alpha={}, t_receiver = {} s and dist ={} with c_esv = {}m/s'.format(x_min, z_min, best_alpha, t_receiver, dist_min, c_esv))

            return c_esv

        # Update previous minimum distance and adjust alpha range for the next iteration
        alpha_range = np.linspace(best_alpha - alpha_step, best_alpha + alpha_step, 100)
        alpha_step /= 2

    # If no ray reaches the receiver, print a message
    if x_min is None or z_min is None:
        # return('No ray reaches the receiver, change angle or tolerance threshold, dist = {}'.format(dist_min))
        return(np.nan)

def process_beta(beta, dz):
    return dist_analytic(dz, beta)

def table_angle(dz, step):
    list_beta = np.arange(10,100, step)
    print(list_beta)
    c_esv = np.zeros_like(list_beta)

    with mp.Pool() as pool:
        results = pool.starmap(process_beta, [(beta, dz) for beta in list_beta])

    c_esv = np.array(results)

    data = {'list_beta': list_beta.astype(float),
    'c_esv': c_esv.astype(float)}

    filename = f'data_{dz}_step_angle_{step}.mat'
    savemat(filename, data)


if __name__ == '__main__':
    table_angle(5226,1)
    print(sio.loadmat('data_5225_step_angle_10.mat'))