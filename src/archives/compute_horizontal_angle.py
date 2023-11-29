import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import math


def load_and_process_data(path):
    data = sio.loadmat(path)
    days = data['days'].flatten() - 59015
    times = data['times'].flatten()
    datetimes = ((days * 24 * 3600) + times) / 3600
    x, y, z = data['x'].flatten(), data['y'].flatten(), data['z'].flatten()
    return datetimes, x, y, z

paths = [
    '../../data/SwiftNav_Data/Unit1-camp_bis.mat',
    '../../data/SwiftNav_Data/Unit2-camp_bis.mat',
    '../../data/SwiftNav_Data/Unit3-camp_bis.mat',
    '../../data/SwiftNav_Data/Unit4-camp_bis.mat'
]

all_data = [load_and_process_data(path) for path in paths]
common_datetimes = set(all_data[0][0])
for data in all_data[1:]:
    common_datetimes.intersection_update(data[0])
common_datetimes = sorted(common_datetimes)

filtered_data = []
for datetimes, x, y, z in all_data:
    mask = np.isin(datetimes, common_datetimes)
    filtered_data.append((datetimes[mask], x[mask], y[mask], z[mask]))

def compute_angle(A, B):
    # Calculer le vecteur entre A et B
    AB = np.array([B[0]-A[0], B[1]-A[1], B[2]-A[2]])

    # Vecteur vertical
    k = np.array([0, 0, 1])

    # Calculer le cosinus de l'angle
    cos_theta = np.dot(AB, k) / np.linalg.norm(AB)
    print(np.degrees(np.arccos(cos_theta)))
    # Retourner l'angle en degrés
    return np.degrees(np.arccos(cos_theta))

def average_angle(A_data, B_data):
    angles = [compute_angle([A_data[1][i], A_data[2][i], A_data[3][i]], [B_data[1][i], B_data[2][i], B_data[3][i]]) for i in range(len(A_data[0]))]
    return np.mean(angles)

angle_A1_A2 = average_angle(filtered_data[0], filtered_data[1])
angle_A3_A4 = average_angle(filtered_data[2], filtered_data[3])
angle_A3_A2 = average_angle(filtered_data[2], filtered_data[1])
angle_A1_A4 = average_angle(filtered_data[0], filtered_data[3])

print("Angle moyen entre A1 et A2:", angle_A1_A2)
print("Angle moyen entre A3 et A4:", angle_A3_A4)
print("Angle moyen entre A3 et A2:", angle_A3_A2)
print("Angle moyen entre A1 et A4:", angle_A1_A4)
