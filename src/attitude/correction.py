import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np

data_unit1 = sio.loadmat('../../data/SwiftNav_Data/GPStime_synchro/Unit1-camp_bis_GPStime_synchro.mat')
data_unit2 = sio.loadmat('../../data/SwiftNav_Data/GPStime_synchro/Unit2-camp_bis_GPStime_synchro.mat')
data_unit3 = sio.loadmat('../../data/SwiftNav_Data/GPStime_synchro/Unit3-camp_bis_GPStime_synchro.mat')
data_unit4 = sio.loadmat('../../data/SwiftNav_Data/GPStime_synchro/Unit4-camp_bis_GPStime_synchro.mat')
#
#
# def plot_antenna_data(unit_number, subplot_num, color):
#     data_unit = sio.loadmat(f'../../data/SwiftNav_Data/Unit{unit_number}-camp_bis.mat')
#     print(data_unit)
#     lat = data_unit['lat'].flatten()
#     lon = data_unit['lon'].flatten()
#     elev = data_unit['elev'].flatten()
#
#     plt.subplot(3, 2, subplot_num)
#     plt.plot(lat, label=f"Latitude Antenna {unit_number}", color=color)
#     plt.xlabel('Time')
#     plt.ylabel('Latitude')
#     plt.legend()
#
#     plt.subplot(3, 2, subplot_num + 2)
#     plt.plot(lon, label=f"Longitude Antenna {unit_number}", color=color)
#     plt.xlabel('Time')
#     plt.ylabel('Longitude')
#     plt.legend()
#
#     plt.subplot(3, 2, subplot_num + 4)
#     plt.plot(elev, label=f"Elevation Antenna {unit_number}", color=color)
#     plt.xlabel('Time')
#     plt.ylabel('Elevation')
#     plt.legend()
#
# plt.figure(figsize=(15, 10))
#
# # Plot pour Antenna 3 et 4
# plot_antenna_data(3, 1, 'r')
# plot_antenna_data(4, 2, 'g')
#
# # Plot pour Antenna 1 et 2
# plot_antenna_data(1, 1, 'b')
# plot_antenna_data(2, 2, 'c')
#
# plt.tight_layout()
# plt.show()
#
#
# def calculate_roll(z1, z2, d):
#     delta_z = z2 - z1
#     roll_rad = np.arctan2(delta_z, d)
#     roll_deg = np.degrees(roll_rad)
#     return roll_deg
#
# # Charger les données pour l'antenne 3
# data_unit3 = sio.loadmat('../../data/SwiftNav_Data/GPStime_synchro/Unit3-camp_bis_GPStime_synchro.mat')
# z3 = data_unit3['z'].flatten()
#
# # Charger les données pour l'antenne 4
# data_unit4 = sio.loadmat('../../data/SwiftNav_Data/GPStime_synchro/Unit4-camp_bis_GPStime_synchro.mat')
# z4 = data_unit4['z'].flatten()
#
# # S'assurer que les deux ensembles de données ont la même taille
# if z3.shape[0] != z4.shape[0]:
#     print("Les deux ensembles de données doivent avoir la même taille!")
#     exit()
#
# # Distance entre les antennes A3 et A4 en mètres
# d = 7.0
#
# # Calculer le roulis pour chaque point temporel
# roll_deg = calculate_roll(z3, z4, d)
#
# # Tracer le roulis au fil du temps
# plt.figure()
# plt.plot(roll_deg)
# plt.xlabel('Point temporel')
# plt.ylabel('Roulis (degrés)')
# plt.title('Roulis estimé entre les antennes 3 et 4')
# plt.show()
#
# def calculate_pitch(z1, z2, d):
#     delta_z = z2 - z1
#     pitch_rad = np.arctan2(delta_z, d)
#     pitch_deg = np.degrees(pitch_rad)
#     return pitch_deg
#
# # Charger les données pour l'antenne 1
# data_unit1 = sio.loadmat('../../data/SwiftNav_Data/GPStime_synchro/Unit1-camp_bis_GPStime_synchro.mat')
# z1 = data_unit1['z'].flatten()
#
# # Charger les données pour l'antenne 4
# data_unit4 = sio.loadmat('../../data/SwiftNav_Data/GPStime_synchro/Unit4-camp_bis_GPStime_synchro.mat')
# z4 = data_unit4['z'].flatten()
#
# # S'assurer que les deux ensembles de données ont la même taille
# if z1.shape[0] != z4.shape[0]:
#     print("Les deux ensembles de données doivent avoir la même taille!")
#     exit()
#
# # Distance entre les antennes A1 et A4 en mètres (à ajuster selon vos données)
# d = 10  # Remplacez par la distance réelle entre A1 et A4
#
# # Calculer le tangage pour chaque point temporel
# pitch_deg = calculate_pitch(z1, z4, d)
#
# # Tracer le tangage au fil du temps
# plt.figure()
# plt.plot(pitch_deg)
# plt.xlabel('Point temporel')
# plt.ylabel('Tangage (degrés)')
# plt.title('Tangage estimé entre les antennes 1 et 4')
# plt.show()
#
# def calculate_yaw(x1, y1, x2, y2):
#     delta_x = x2 - x1
#     delta_y = y2 - y1
#     yaw_rad = np.arctan2(delta_y, delta_x)
#     yaw_deg = np.degrees(yaw_rad)
#     return yaw_deg
#
# # Charger les données pour l'antenne 2
# data_unit2 = sio.loadmat('../../data/SwiftNav_Data/GPStime_synchro/Unit2-camp_bis_GPStime_synchro.mat')
# x2 = data_unit2['x'].flatten()
# y2 = data_unit2['y'].flatten()
#
# # Charger les données pour l'antenne 4
# data_unit4 = sio.loadmat('../../data/SwiftNav_Data/GPStime_synchro/Unit4-camp_bis_GPStime_synchro.mat')
# x4 = data_unit4['x'].flatten()
# y4 = data_unit4['y'].flatten()
#
# # S'assurer que les deux ensembles de données ont la même taille
# if x2.shape[0] != x4.shape[0]:
#     print("Les deux ensembles de données doivent avoir la même taille!")
#     exit()
#
# # Calculer le lacet pour chaque point temporel
# yaw_deg = calculate_yaw(x2, y2, x4, y4)
#
# # Tracer le lacet au fil du temps
# plt.figure()
# plt.plot(yaw_deg)
# plt.xlabel('Point temporel')
# plt.ylabel('Lacet (degrés)')
# plt.title('Lacet estimé entre les antennes 2 et 4')
# plt.show()


import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

def scatter_single_data(time, data, label, xlabel, ylabel, color):
    plt.scatter(time, data,s=10, label=label, color=color)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()

def calculate_angle(z1, z2, d):
    delta_z = z2 - z1
    angle_rad = np.arctan2(delta_z, d)
    angle_deg = np.degrees(angle_rad)
    return angle_deg

pairs = [(3, 4, 7.0, 'Roulis'), (1, 4, 10.0, 'Tangage'), (2, 4, 5.0, 'Lacet')]
colors = ['r', 'g', 'b', 'c']
data_types = ['x', 'y', 'z']

for unit1, unit2, d, angle_type in pairs:
    plt.figure(figsize=(15, 10))

    for i, data_type in enumerate(data_types):
        plt.subplot(4, 1, i+1)

        data_unit1 = sio.loadmat(f'../../data/SwiftNav_Data/GPStime_synchro/Unit{unit1}-camp_bis_GPStime_synchro.mat')
        data_unit2 = sio.loadmat(f'../../data/SwiftNav_Data/GPStime_synchro/Unit{unit2}-camp_bis_GPStime_synchro.mat')

        days = data_unit1['days'].flatten() - 59015
        times = data_unit1['times'].flatten()
        datetimes = (days * 24 * 3600) + times
        condition_gnss = (datetimes / 3600 >= 25) & (datetimes / 3600 <= 40.9)
        time_GNSS = datetimes[condition_gnss] / 3600

        data = data_unit1[data_type].flatten()[condition_gnss]
        scatter_single_data(time_GNSS, data, f"{data_type.capitalize()} Antenna {unit1}", 'Time', f'{data_type.capitalize()}', colors[unit1-1])

        days = data_unit2['days'].flatten() - 59015
        times = data_unit2['times'].flatten()
        datetimes = (days * 24 * 3600) + times
        condition_gnss = (datetimes / 3600 >= 25) & (datetimes / 3600 <= 40.9)
        time_GNSS = datetimes[condition_gnss] / 3600

        data = data_unit2[data_type].flatten()[condition_gnss]
        scatter_single_data(time_GNSS, data, f"{data_type.capitalize()} Antenna {unit2}", 'Time', f'{data_type.capitalize()}', colors[unit2-1])

    plt.subplot(4, 1, 4)
    z1 = data_unit1['z'].flatten()[condition_gnss]
    z2 = data_unit2['z'].flatten()[condition_gnss]
    angle_deg = calculate_angle(z1, z2, d)
    scatter_single_data(time_GNSS, angle_deg, f'{angle_type} entre A{unit1} et A{unit2}', 'Time', f'{angle_type} (degrés)', 'm')

    plt.tight_layout()
    plt.show()

# Fonction pour calculer le roulis, le tangage et le lacet
def calculate_angle(z1, z2, x1, x2, y1, y2, d):
    delta_z = z2 - z1
    delta_x = x2 - x1
    delta_y = y2 - y1

    roll_rad = np.arctan2(delta_z, d)
    roll_deg = np.degrees(roll_rad)

    pitch_rad = np.arctan2(delta_z, d)
    pitch_deg = np.degrees(pitch_rad)

    yaw_rad = np.arctan2(delta_y, delta_x)
    yaw_deg = np.degrees(yaw_rad)

    return roll_deg, pitch_deg, yaw_deg


# Calcul des angles
roll_deg_34, _, _ = calculate_angle(data_unit3['z'].flatten(), data_unit4['z'].flatten(), None, None, None, None, 7.0)
_, pitch_deg_14, _ = calculate_angle(data_unit1['z'].flatten(), data_unit4['z'].flatten(), None, None, None, None, 10.0)
_, _, yaw_deg_24 = calculate_angle(None, None, data_unit2['x'].flatten(), data_unit4['x'].flatten(), data_unit2['y'].flatten(), data_unit4['y'].flatten(), None)

# Création de la figure et des sous-tracés
plt.figure(figsize=(15, 15))

# Tracé du roulis
plt.subplot(3, 1, 1)
plt.plot(roll_deg_34, label='Roulis entre 3 et 4')
plt.xlabel('Point temporel')
plt.ylabel('Roulis (degrés)')
plt.legend()

# Tracé du tangage
plt.subplot(3, 1, 2)
plt.plot(pitch_deg_14, label='Tangage entre 1 et 4')
plt.xlabel('Point temporel')
plt.ylabel('Tangage (degrés)')
plt.legend()

# Tracé du lacet
plt.subplot(3, 1, 3)
plt.plot(yaw_deg_24, label='Lacet entre 2 et 4')
plt.xlabel('Point temporel')
plt.ylabel('Lacet (degrés)')
plt.legend()

plt.tight_layout()
plt.show()
