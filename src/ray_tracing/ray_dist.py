import numpy as np
import scipy.io as sio
import math
from numeric_sol import *
from analytic_sol import *
from scipy.optimize import minimize
from tqdm import tqdm

def dist_numeric(xyzS, xyzR):
    '''
    Inputs :
        xyzS : position of the source
        xyzR : position of the receiver
    Outputs :
        dist : distance between source and receiver
        time : travel time
        alpha : initial ray angle
    '''

    # Extract coordinates of source and receiver
    xs, ys, zs = xyzS[0], xyzS[1], xyzS[2]
    xr, yr, zr = xyzR[0], xyzR[1], xyzR[2]

    # Set initial parameters
    alpha_range = np.linspace(0, 90, 100)
    alpha_step = (max(alpha_range) - min(alpha_range)) / len(alpha_range)
    tolerance = 0.01

    #Parameters of time integration
    tMax, dt = 10, 0.001

    # Initialize variables for tracking minimum distance and associated parameters
    dist_min = np.inf
    x_min, z_min = None, None
    t_receiver = None
    best_alpha = None
    x_fin = np.empty(0)
    z_fin = np.empty(0)

    # Set flags for plotting and early termination
    plot_results = True
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
            x, z = ray_numeric(angle, zs, xs, tMax, dt)
            t = np.arange(0, tMax, dt)

            # Calculate errors and find minimum error index
            errors = np.sqrt((xr - x) ** 2 + (zr - z) ** 2)
            errors_all[i] = np.nanmin(errors)
            id_min = np.nanargmin(errors)

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

        # Check for early termination or increasing minimum distance
        if early_termination :
            if plot_results:
                # Plot the results if plot_results flag is set
                plt.scatter(xs, zs, color='green', label='Source')
                plt.scatter(xr, zr, color='red', label='Receiver')
                plt.plot(x_fin, z_fin, label='Alpha = {}°'.format(best_alpha))
                plt.gca().invert_yaxis()
                plt.title("Ray Path in Linear Wavespeed profile")
                plt.xlabel("Horizontal Distance (km)")
                plt.ylabel("Depth (km)")
                plt.legend()
                plt.show()
            # Print the calculated values and return distance and time
            print('xcalc={} and zcalc={} for alpha={}, t_receiver = {} s and dist ={}'.format(x_min, z_min, best_alpha, t_receiver, dist_min))
            return dist_min, t_receiver, best_alpha

        # Update previous minimum distance and adjust alpha range for the next iteration
        prev_dist_min = dist_min
        alpha_range = np.linspace(best_alpha - alpha_step, best_alpha + alpha_step, 100)
        alpha_step /= 2

    # If no ray reaches the receiver, print a message
    if x_min is None or z_min is None:
        print('No ray reaches the receiver, change angle or tolerance threshold, dist = {}'.format(dist_min))
    # Return the calculated values
    return x_min, z_min, best_alpha, t_receiver, dist_min

def dist_analytic(xyzS, xyzR):
    '''
    Inputs :
        xyzS : position of the source
        xyzR : position of the receiver
    Outputs :
        dist : distance between source and receiver
        time : travel time
        alpha : initial ray angle
    '''

    # Extract coordinates of source and receiver
    xs, ys, zs = xyzS[0], xyzS[1], xyzS[2]
    xr, yr, zr = xyzR[0], xyzR[1], xyzR[2]

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
            # Add the initial value of the source
            x = xs + x


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

        # Check for early termination or increasing minimum distance
        if early_termination :
            if plot_results:
                # Plot the results if plot_results flag is set
                plt.scatter(xs, zs, color='green', label='Source')
                plt.scatter(xr, zr, color='red', label='Receiver')
                plt.plot(x_fin, z_fin, label='Alpha = {}°'.format(best_alpha))
                plt.gca().invert_yaxis()
                plt.title("Ray Path in Linear Wavespeed profile")
                plt.xlabel("Horizontal Distance (km)")
                plt.ylabel("Depth (km)")
                plt.legend()
                plt.show()
            # Print the calculated values and return distance and time
            print('xcalc={} and zcalc={} for alpha={}, t_receiver = {} s and dist ={}'.format(x_min, z_min, best_alpha, t_receiver, dist_min))
            print("Final alpha_range: ", alpha_range)
            return dist_min, t_receiver, best_alpha

        # Update previous minimum distance and adjust alpha range for the next iteration
        alpha_range = np.linspace(best_alpha - alpha_step, best_alpha + alpha_step, 100)
        alpha_step /= 2

    # If no ray reaches the receiver, print a message
    if x_min is None or z_min is None:
        return('No ray reaches the receiver, change angle or tolerance threshold, dist = {}'.format(dist_min))

    # Return the calculated values
    return best_alpha, t_receiver, dist_min


if __name__ == '__main__':
    # sv=np.loadtxt('../../data/sv_GDEM.txt')
    # depth=np.loadtxt('../../data/depth_GDEM.txt')
    # Paramètres xyzS et xyzR
    xyzS = [2000, 1500, 0]
    xyzR = [10000, 3000, 5225]

    dist_analytic(xyzS, xyzR)
    # dist_numeric(xyzS, xyzR)
