import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.interpolate import interp1d
from numba import jit
from scipy.interpolate import CubicSpline

cz = np.loadtxt('../../data/cz_interp5250.txt')
depth = np.loadtxt('../../data/depth_interp5250.txt')
gradc = np.loadtxt('../../data/gradc_interp5250.txt')

# sv=np.loadtxt('../../data/sv_GDEM.txt')
# depthg=np.loadtxt('../../data/depth_GDEM.txt')
#
# def preprocess_data(cz, depth):
#     z = np.linspace(0, np.max(depth) + 50, int(1e6)+4784)
#     f = interp1d(depth, cz, 'linear', bounds_error=False, fill_value='extrapolate')
#     cz = f(z)
#     depth = z
#     gradc = np.diff(cz) / np.diff(depth)
#     gradc = np.insert(gradc, 0, gradc[0])
#     return cz, depth, gradc
#
# cz2,depth2,gradc2 = preprocess_data(sv,depthg)
# np.savetxt('cz_interp5250.txt',cz2)
# np.savetxt('depth_interp5250.txt',depth2)
# np.savetxt('gradc_interp5250.txt',gradc2)

def ray_analytic(cz, depth, gradc, theta0):
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
    return x, depth, t


if __name__=='__main__':
    cz = np.loadtxt('../../data/cz_interp.txt')
    depth = np.loadtxt('../../data/depth_interp.txt')
    gradc = np.loadtxt('../../data/gradc_interp.txt')

    # sv=np.loadtxt('../../data/sv_GDEM.txt')
    # depthg=np.loadtxt('../../data/depth_GDEM.txt')
    #
    # plt.scatter(gradc[-10000:-9500], depth[-10000:-9500] , label = "Avec interp")
    # # plt.scatter(sv,depthg, label = "Sans Interp")
    # plt.xlabel('Vitesse du son en m/s')
    # plt.ylabel('Profondeur en m')
    # plt.gca().invert_yaxis()
    # plt.legend()
    # plt.show()

    # data = sio.loadmat('data_5225_step_angle_2.mat')
    # esv = data['c_esv']
    # angle = data['list_beta']
    # plt.scatter(angle,esv, label='5225 m')
    # data = sio.loadmat('data_5227_step_angle_2.mat')
    # esv = data['c_esv']
    # angle = data['list_beta']
    # plt.scatter(angle,esv, label='5227 m')
    # data = sio.loadmat('data_5223_step_angle_2.mat')
    # esv = data['c_esv']
    # angle = data['list_beta']
    # plt.scatter(angle,esv, label='5223 m')
    # data = sio.loadmat('data_5220_step_angle_2.mat')
    # esv = data['c_esv']
    # angle = data['list_beta']
    # plt.scatter(angle,esv, label='5220 m')
    # data = sio.loadmat('data_5200_step_angle_2.mat')
    # esv = data['c_esv']
    # angle = data['list_beta']
    # plt.scatter(angle,esv, label='5200 m')
    # plt.legend()
    # plt.xlabel("Angle d'élévation en degrés")
    # plt.ylabel("Vitesse équivalente en m/s")
    #
    # plt.show()

    # x, z, t=ray_analytic(cz,depth,gradc,45)
    # plt.plot(x,z)
    # plt.show()
