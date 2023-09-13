import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.interpolate import interp1d
from numba import jit
from scipy.interpolate import CubicSpline

data = np.loadtxt('../../dat/cast2.txt')

# Charger les données (simulées ici pour l'exemple)
data = np.loadtxt('../../dat/cast2.txt')
depth = data[:, 0]
sv = data[:, 1]

# Trouver l'indice où depth est maximale
max_depth_index = np.argmax(depth)
print(np.max(depth))
import sys
sys.exit()
# Utiliser seulement les données jusqu'à max_depth_index
depth_descending = depth[:max_depth_index + 1]
sv_descending = sv[:max_depth_index + 1]

# Générer les indices pour 5200 points régulièrement espacés (ou moins si le tableau est plus petit)
num_points = 100
indices = np.linspace(0, len(depth_descending) - 1, num_points, dtype=int)

# Extraire les valeurs correspondantes de depth et sv
depth_sampled = depth_descending[indices]
sv_sampled = sv_descending[indices]

# Tracer les données
plt.figure()
plt.plot(sv_sampled, depth_sampled)
plt.xlabel('sv')
plt.ylabel('depth')
plt.title('Plot de sv et depth')
plt.gca().invert_yaxis()  # Inverser l'axe des y si nécessaire
plt.show()



def preprocess_data(cz, depth):
    z = np.linspace(0, 5200 + 50, int(1e6)+4784)
    f = interp1d(depth, cz, 'cubic', fill_value='extrapolate')
    cz = f(z)
    depth = z
    gradc = np.diff(cz) / np.diff(depth)
    gradc = np.insert(gradc, 0, gradc[0])
    return cz, depth, gradc

cz,depth,gradc = preprocess_data(sv_sampled, depth_sampled)

# Tracer les données
plt.figure()
plt.plot(cz, depth)
plt.xlabel('sv')
plt.ylabel('depth')
plt.title('Plot de sv et depth')
plt.gca().invert_yaxis()  # Inverser l'axe des y si nécessaire
plt.show()

np.savetxt('../../data/cz_cast2_big.txt',cz)
np.savetxt('../../data/depth_cast2_big.txt',depth)
np.savetxt('../../data/gradc_cast2_big.txt',gradc)

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

# Création du graphique
plt.figure()

# Boucle sur les angles de 0 à 90 degrés par incréments de 10
for angle in range(0, 91, 10):
    x, z, t = ray_analytic(cz, depth, gradc, angle)
    plt.plot(x, z, label=f'Angle = {angle}°')

# Inverser l'axe des z
plt.gca().invert_yaxis()


# Ajouter des étiquettes et une légende
plt.xlabel('x')
plt.ylabel('z')
plt.legend()

# Afficher le graphique
plt.show()
print(t)
if __name__=='__main__':
    data = np.loadtxt('../../dat/cast2.txt')



    sv=np.loadtxt('../../data/SV/cz_cast2.txt')
    depthg=np.loadtxt('../../data/SV/depth_cast2.txt')
    #
    # plt.scatter(gradc[-10000:-9500], depth[-10000:-9500] , label = "Avec interp")
    # # plt.scatter(sv,depthg, label = "Sans Interp")
    # plt.xlabel('Vitesse du son en m/s')
    # plt.ylabel('Profondeur en m')
    # plt.gca().invert_yaxis()
    # plt.legend()
    # plt.show()
