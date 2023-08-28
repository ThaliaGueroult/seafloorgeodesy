import numpy as np
import scipy.io as sio
from scipy.io import savemat
import math
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from tqdm import tqdm
from analytic_sol import *
import multiprocessing as mp

def dist_analytic(dz, beta):
    '''
    Inputs :
        dz : vertical distance between source and receiver
        beta : elevation angle between source and receiver
    Outputs :
        c_esv : effective sound velocity between source and receiver
    '''

    # Compute horizontal distance between source and receiver
    beta = beta * np.pi / 180
    dh = dz / np.tan(beta)
    xr, zr = dh, dz

    # Set initial parameters
    alpha_range = np.linspace(0, 90, 100)
    alpha_step = (max(alpha_range) - min(alpha_range)) / len(alpha_range)

    # Initialize variables for tracking minimum distance and associated parameters
    dist_min = np.inf
    x_min, z_min = None, None
    t_receiver = None
    best_alpha = None

    while alpha_step > 10e-10:
        alpha = np.array(alpha_range)

        errors_all = np.empty((len(alpha),))
        x_all = np.empty((len(alpha),))
        z_all = np.empty((len(alpha),))
        t_all = np.empty((len(alpha),))

        for i, angle in enumerate(alpha):
            # Calculate ray path for the given angle
            x, z, t = ray_analytic(cz, depth, gradc, angle)

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

        # Update previous minimum distance and adjust alpha range for the next iteration
        alpha_range = np.linspace(best_alpha - alpha_step, best_alpha + alpha_step, 100)
        alpha_step /= 2

    # Calculate effective sound velocity
    if beta == 0:
        d_sr = dz
        c_esv = dz / t_receiver
    else:
        d_sr = dz / np.sin(beta)
        c_esv = d_sr / t_receiver
        print('xcalc={} and zcalc={} for alpha={}, t_receiver = {} s and dist ={} with c_esv = {}m/s'.format(x_min, z_min, best_alpha, t_receiver, dist_min, c_esv))

    return c_esv

def process_beta(beta, dz):
    return dist_analytic(dz, beta)

def table_angle(dz, step):
    list_beta = np.arange(45,90,step)
    print(list_beta)
    c_esv = np.zeros_like(list_beta)

    with mp.Pool() as pool:
        results = pool.starmap(process_beta, [(beta, dz) for beta in list_beta])
        print(results)

    c_esv = np.array(results)

    data = {'list_beta': list_beta.astype(float),
    'c_esv': c_esv.astype(float)}

    filename = f'esv_table_without_tol/data_{dz}_step_angle_{step}.mat'
    savemat(filename, data)

def plot_esv():

    # Numéros de fichiers de 5226 à 5249
    file_numbers = range(5182, 5197)

    # Initialisation des listes pour les données des fichiers
    all_angles = []
    all_esv = []

    # Liste de labels et de couleurs
    labels = []
    colors = plt.cm.viridis(np.linspace(0, 1, len(file_numbers)))

    for file_number in file_numbers:
        # Nom du fichier
        file_name = f"esv_table_without_tol/data_{file_number}_step_angle_2.mat"

        # Charger les données depuis le fichier
        data = sio.loadmat(file_name)
        esv = data['c_esv']
        angle = data['list_beta']

        # Ajouter les données à la liste globale
        all_angles.extend(angle)
        all_esv.extend(esv)

        # Créer un label unique pour chaque fichier
        label = f'{file_number} m'
        labels.append(label)

    # Plot des données avec des labels et des couleurs différents pour chaque fichier
    for i in range(len(file_numbers)):
        plt.scatter(all_angles[i], all_esv[i], label=labels[i], color=colors[i])

    plt.legend()
    plt.title('Effective Sound Velocity')
    plt.xlabel('Angle')
    plt.ylabel('ESV')
    plt.show()

def plot_and_save_esv():
    # Numéros de fichiers de 5226 à 5249
    file_numbers = range(5250,5251)

    # Liste de labels et de couleurs
    labels = []
    colors = plt.cm.rainbow(np.linspace(0, 1, len(file_numbers)))


    for i, file_number in enumerate (file_numbers):
        # Nom du fichier
        file_name = f"esv_table_without_tol/data_{file_number}_step_angle_2.mat"

        # Ajouter le label et la couleur du fichier
        labels.append(f"File {file_number}")

        # Charger les données depuis le fichier
        data = sio.loadmat(file_name)
        esv = data['c_esv']
        angle = data['list_beta']

        # Interpolation de 45 à 90 degrés avec un step de 0.01 degrés
        new_angles = np.arange(45, 90, 0.01)
        f = interp1d(angle.flatten(), esv.flatten(), kind='cubic',bounds_error=False, fill_value='extrapolate')
        new_esv = f(new_angles)

        # Enregistrer les données interpolées dans un fichier avec le nom reflétant le step de 0.01
        new_file_name = f"esv_table_without_tol/mat_interp/data_{file_number}_step_angle_0_01.mat"
        sio.savemat(new_file_name, {'elevation angle (deg)': new_angles, 'esv (m/s)': new_esv})

        plt.plot(new_angles, new_esv, label=labels[i], color=colors[i])

    plt.xlabel('Angle (degrees)')
    plt.ylabel('ESV')
    plt.title('Interpolated ESV data')
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=0., ncol = 2)
    plt.show()


def compare_interpolations(file_number=5225):
    # Charger les données depuis le fichier
    file_name = f"esv_table_without_tol/data_{file_number}_step_angle_2.mat"
    data = sio.loadmat(file_name)
    esv = data['c_esv']
    angle = data['list_beta']

    # Interpolation de 45 à 90 degrés avec un step de 0.01 degrés
    new_angles = np.arange(45, 90, 0.01)

    # Liste des méthodes d'interpolation possibles
    # interpolation_methods = ['linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic']
    interpolation_methods = ['linear', 'nearest', 'quadratic', 'cubic']
    colors = plt.cm.rainbow(np.linspace(0, 1, len(interpolation_methods)))

    plt.figure(figsize=(10, 6))
    plt.scatter(angle, esv, label='Original Data', color='red', marker='x')

    for method, color in zip(interpolation_methods, colors):
        f = interp1d(angle.flatten(), esv.flatten(), kind=method, bounds_error=False, fill_value='extrapolate')
        new_esv = f(new_angles)
        plt.plot(new_angles, new_esv, label=f'{method.capitalize()} Interpolation', color=color)

    plt.xlabel('Angle (degrees)')
    plt.ylabel('ESV')
    plt.title(f'Interpolation Comparison for File {file_number}')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    # table_angle(5130,2)
    # table_angle(5250,2)
    # for k in range (8):
    #     table_angle(5250+k,2)
    # plot_esv()
    plot_and_save_esv()

    compare_interpolations(file_number=5225)
