import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.interpolate import interp1d
from numba import jit
from scipy.interpolate import CubicSpline

data = np.loadtxt('../../dat/cast2.txt')

depth = data[:, 0]
sv = data[:, 1]

# Trouver l'indice où la profondeur atteint sa valeur maximale (correspondant à la descente)
indice_max_profondeur = depth.argmax()

# Extraire uniquement les données jusqu'à l'indice de la profondeur maximale (descente)
depth = depth[:indice_max_profondeur + 1]
sv = sv[:indice_max_profondeur + 1]

# Trier depth_descente par ordre croissant et obtenir les indices de tri
indices_tri = np.argsort(depth)

# Trier depth_descente et sv_descente en utilisant les indices de tri
depth_t = depth[indices_tri]
sv_t = sv[indices_tri]

indices_uniques = np.unique(depth_t, return_index=True)[1]

# Utiliser les indices uniques pour obtenir les profondeurs et vitesses correspondantes sans doublons
depth_tu = depth_t[indices_uniques]
sv_tu = sv_t[indices_uniques]

# Appliquer un lissage à l'aide d'une moyenne mobile avec une fenêtre de taille 5
window_size = 100
sv_tul = np.convolve(sv_tu, np.ones(window_size)/window_size, mode='valid')

# Créer un vecteur pour les profondeurs correspondant aux données lissées
depth_tul = depth_tu[window_size//2:-(window_size//2)]


def preprocess_data(cz, depth):
    z = np.linspace(0, 5200 + 50, int(1e6)+4784)
    f = interp1d(depth, cz, 'quadratic', fill_value='extrapolate')
    cz = f(z)
    depth = z
    gradc = np.diff(cz) / np.diff(depth)
    gradc = np.insert(gradc, 0, gradc[0])
    return cz, depth, gradc

cz2,depth2,gradc2 = preprocess_data(sv_tu,depth_tu)
plt.plot(cz2,depth2)
plt.show()

np.savetxt('../../data/cz_cast2.txt',cz2)
np.savetxt('../../data/depth_cast2.txt',depth2)
np.savetxt('../../data/gradc_cast2.txt',gradc2)

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
    data = np.loadtxt('../../dat/cast2.txt')


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
