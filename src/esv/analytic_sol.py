import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.interpolate import interp1d
from numba import jit
from scipy.interpolate import CubicSpline
import time

cz = np.loadtxt('../../data/SV/cz_cast2_big.txt')
depth = np.loadtxt('../../data/SV/depth_cast2_big.txt')
gradc = np.loadtxt('../../data/SV/gradc_cast2_big.txt')

# cz = np.loadtxt('../../data/SV/cz_interp5250.txt')
# depth = np.loadtxt('../../data/SV/depth_interp5250.txt')
# gradc = np.loadtxt('../../data/SV/gradc_interp5250.txt')
#
# sv=np.loadtxt('../../data/SV/sv_GDEM.txt')
# depthg=np.loadtxt('../../data/SV/depth_GDEM.txt')
#
# def preprocess_data(cz, depth):
#     z = np.linspace(0, np.max(depth) + 50, 3000)
#     f = interp1d(depth, cz, 'linear', bounds_error=False, fill_value='extrapolate')
#     cz = f(z)
#     depth = z
#     gradc = np.diff(cz) / np.diff(depth)
#     gradc = np.insert(gradc, 0, gradc[0])
#     return cz, depth, gradc
#
# cz2,depth2,gradc2 = preprocess_data(sv,depthg)
# np.savetxt('../../data/SV/cz_interp3000.txt',cz2)
# np.savetxt('../../data/SV/depth_interp3000.txt',depth2)
# np.savetxt('../../data/SV/gradc_interp3000.txt',gradc2)

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

def preprocess_data(cz, depth):
    z = np.linspace(0, 5200 + 50, int(1e6)+4784)
    f = interp1d(depth, cz, 'cubic', fill_value='extrapolate')
    cz = f(z)
    depth = z
    gradc = np.diff(cz) / np.diff(depth)
    gradc = np.insert(gradc, 0, gradc[0])
    return cz, depth, gradc


if __name__=='__main__':
    cz = np.loadtxt('../../data/SV/cz_cast2_big.txt')
    depth = np.loadtxt('../../data/SV/depth_cast2_big.txt')
    gradc = np.loadtxt('../../data/SV/gradc_cast2_big.txt')
    # cz = np.loadtxt('../../data/SV/cz_interp5250.txt')
    # depth = np.loadtxt('../../data/SV/depth_interp5250.txt')
    # gradc = np.loadtxt('../../data/SV/gradc_interp5250.txt')

    # t = time.time()
    # alpha = np.linspace(0, 90, 10) # degrees below Horizontal
    # fig, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [2, 1]})
    # # Plot the ray paths
    # for k, alpha_k in enumerate(alpha):
    #     xPath, zPath, tPath = ray_analytic(cz, depth, gradc, alpha_k)
    #     ax1.plot(xPath, zPath, label='{}°'.format(alpha_k))
    # # Plot the wavespeed profile
    # ax2.plot(cz, depth)
    # # Configure the plot settings
    # # ax1.scatter(22761, 5025, color='red', label='Receiver')
    # ax1.set_xlim(0, 32000)
    # ax1.set_ylim(np.min(depth), np.max(depth))
    # ax1.invert_yaxis()
    # ax1.legend()
    # ax2.set_ylim(np.min(depth), np.max(depth))
    # ax2.invert_yaxis()
    # ax1.set_title("Ray Path analytic in Linear Wavespeed profile")
    # ax1.set_xlabel("Horizontal Distance (km)")
    # ax1.set_ylabel("Depth (km)")
    # ax2.set_title("Sound Celerity Profile Bermuda")
    # ax2.set_xlabel("Sound Celerity (km/s)")
    # ax2.set_ylabel("Depth (km)")
    # fig.set_size_inches(12, 4)
    # fig.set_dpi(100)
    # print('time',time.time()-t)
    # plt.show()
    #
    # # Création du graphique
    # plt.figure()
    #
    # # Boucle sur les angles de 0 à 90 degrés par incréments de 10
    # t = time.time()
    # for angle in range(0, 91, 10):
    #     x, z, t = ray_analytic(cz, depth, gradc, angle)
    #     plt.plot(x, z, label=f'Angle = {angle}°')
    # print(time.time()-t)
    # # Inverser l'axe des z
    # plt.gca().invert_yaxis()
    #
    # # Ajouter des étiquettes et une légende
    # plt.xlabel('x')
    # plt.ylabel('z')
    # plt.legend()
    #
    # # Afficher le graphique
    # plt.show()


    # Load the new depth and soundspeed data
    depth_GDEM = np.loadtxt('../../data/SV/depth_GDEM.txt')
    cz_GDEM = np.loadtxt('../../data/SV/sv_GDEM.txt')

    cz_GDEM,depth_GDEM,gradc_GDEM = preprocess_data(cz_GDEM, depth_GDEM)

    # Calculate ray paths with original and new sound celerity profiles for different angles
    angles = [20, 45, 80]
    plt.figure(figsize=(16, 6))

    for angle in angles:
        # Calculate ray paths with original profile
        x_analytic_original, z_analytic_original, _ = ray_analytic(cz, depth, gradc, angle)

        # Calculate ray paths with new profile
        x_analytic_interp, z_analytic_interp, _ = ray_analytic(cz_GDEM, depth_GDEM, gradc_GDEM, angle)

        # Calculate differences between ray paths
        diff = x_analytic_interp - x_analytic_original

        # Plotting ray paths and differences
        plt.subplot(1, 2, 1)
        plt.plot(x_analytic_original, z_analytic_original, label=f'Bermuda - {angle}°',linestyle='-', linewidth=5, alpha=0.5)
        plt.plot(x_analytic_interp, z_analytic_interp, label=f'GDEM - {angle}°', linestyle='--', alpha=1)
        plt.xlabel('Horizontal Distance (x)')
        plt.ylabel('Depth (m)')
        plt.title('Ray Paths with Different Sound Celerity Profiles')
        plt.gca().invert_yaxis()
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(diff, z_analytic_interp, label=f'Angle {angle}°')
        plt.xlabel('Ray Path Difference (m)')
        plt.ylabel('Depth (m)')
        plt.title('Differences between Ray Paths')
        plt.legend()
        plt.gca().invert_yaxis()

    plt.tight_layout()
    plt.show()
