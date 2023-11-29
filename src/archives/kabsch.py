import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from pymap3d import ecef2geodetic


# Load data
data_unit1 = sio.loadmat('../../data/SwiftNav_Data/GPStime_synchro/Unit1-camp_bis_GPStime_synchro.mat')
data_unit2 = sio.loadmat('../../data/SwiftNav_Data/GPStime_synchro/Unit3-camp_bis_GPStime_synchro.mat')
data_unit3 = sio.loadmat('../../data/SwiftNav_Data/GPStime_synchro/Unit1-camp_bis_GPStime_synchro.mat')
data_unit4 = sio.loadmat('../../data/SwiftNav_Data/GPStime_synchro/Unit3-camp_bis_GPStime_synchro.mat')

x1, y1, z1 = data_unit1['x'].flatten(), data_unit1['y'].flatten(), data_unit1['z'].flatten()
x2, y2, z2 = data_unit2['x'].flatten(), data_unit2['y'].flatten(), data_unit2['z'].flatten()
x3, y3, z3 = data_unit3['x'].flatten(), data_unit3['y'].flatten(), data_unit3['z'].flatten()
x4, y4, z4 = data_unit4['x'].flatten(), data_unit4['y'].flatten(), data_unit4['z'].flatten()

A1 = np.vstack((x1, y1, z1)).T
A2 = np.vstack((x2, y2, z2)).T
A3 = np.vstack((x3, y3, z3)).T
A4 = np.vstack((x4, y4, z4)).T

mask = ~np.isnan(A1).any(axis=1) & ~np.isnan(A2).any(axis=1) & ~np.isnan(A4).any(axis=1) & ~np.isnan(A3).any(axis=1)
A1 = A1[mask]
A2 = A2[mask]
A3 = A3[mask]
A4 = A4[mask]

def deg2rad(deg):
    return deg * (np.pi / 180.0)

def kabsch_umeyama(A, B):
    assert A.shape == B.shape
    n, m = A.shape

    EA = np.mean(A, axis=0)
    EB = np.mean(B, axis=0)
    VarA = np.mean(np.linalg.norm(A - EA, axis=1) ** 2)

    H = ((A - EA).T @ (B - EB)) / n
    U, D, VT = np.linalg.svd(H)
    d = np.sign(np.linalg.det(U) * np.linalg.det(VT))
    S = np.diag([1] * (m - 1) + [d])

    R = U @ S @ VT
    c = VarA / np.trace(np.diag(D) @ S)
    t = EA - c * R @ EB

    return R, c, t

def matrix_orthonormalization(C):
    c01 = C[0,:].dot(C[1,:]) / 2.
    c02 = C[0,:].dot(C[2,:]) / 2.
    c12 = C[1,:].dot(C[2,:]) / 2.

    C = np.stack((C[0,:] - c01 * C[1,:] - c02 * C[2,:],
                  C[1,:] - c01 * C[0,:] - c12 * C[2,:],
                  C[2,:] - c02 * C[0,:] - c12 * C[1,:]), 0)

    Cn = np.sqrt((C ** 2).sum(-1))
    C /= Cn[:, None]

    return C

def matrix_to_Tait_Bryan(C):
    C_ = matrix_orthonormalization(C)
    att = np.stack(
        (np.arctan2(C_[...,1, 0], C_[..., 0, 0]),
         np.arctan2(-C_[...,2, 0], np.sqrt(C_[..., 2, 1]**2 + C_[..., 2, 2]**2)),
         np.arctan2(C_[..., 2, 1], C_[..., 2, 2])), -1)
    return att

def get_C_n_e(lonlath):
    """ Calcul de la matrice de passage navigation
    (NED) vers ECEF
    à partir des coordonnées géographiques lonlath
    """
    lon, lat = lonlath
    lon = deg2rad(lon)
    lat = deg2rad(lat)
    # Matrice navigation vers ECEF
    cp = np.cos(lat)
    sp = np.sin(lat)
    cl = np.cos(lon)
    sl = np.sin(lon)

    C_n_e = np.empty((3,3), np.float64)
    C_n_e[...,0,0] = -sp * cl
    C_n_e[...,0,1] = -sl
    C_n_e[...,0,2 ]= -cp * cl
    C_n_e[...,1,0] = -sp * sl
    C_n_e[...,1,1] =  cl
    C_n_e[...,1,2] = -cp * sl
    C_n_e[...,2,0] =  cp
    C_n_e[...,2,1] =  0.
    C_n_e[...,2,2] = -sp
    return C_n_e

def compute_rotations_for_all_times(A1, A2, A3, A4):
    n = len(A1)
    rotations = []

    for t in range(n):
        # Calcul du point de référence comme étant le centre des 4 antennes
        G = (A1[t] + A2[t] + A3[t] + A4[t]) / 4

        # Convertir G (qui est en ECEF) en lonlath
        lat, lon, h = ecef2geodetic(G[0], G[1], G[2])

        lonlath_reference = (lon, lat)
        # Obtenez la matrice de passage NED pour ce point de référence
        C_ne = get_C_n_e(lonlath_reference)
        # print(C_ne)
        # Conversion des antennes en coordonnées NED par rapport à G
        L1 = C_ne @ (A1[t] - G)
        L2 = C_ne @ (A2[t] - G)
        L3 = C_ne @ (A3[t] - G)
        L4 = C_ne @ (A4[t] - G)

        local_coords = np.vstack([L1, L2, L3, L4])
        global_coords = np.vstack([A1[t], A2[t], A3[t], A4[t]])
        R, c, t = kabsch_umeyama(local_coords, global_coords)
        rotations.append(R)

    return rotations

rotations = compute_rotations_for_all_times(A1, A2, A3, A4)
attitudes = np.array([matrix_to_Tait_Bryan(C) for C in rotations])
attitudes = np.degrees(attitudes)

days = data_unit1['days'].flatten() - 59015
times = data_unit1['times'].flatten()
datetimes = ((days * 24 * 3600) + times)/3600

times = datetimes[mask]

plt.figure(figsize=(15, 5))

plt.subplot(3, 1, 1)
plt.scatter(times, attitudes[:, 0],s=1)
plt.title('Angle 1 vs Temps')
plt.xlabel('Temps')
# plt.xlim(29,29.1)
plt.ylabel('Angle 1')

plt.subplot(3, 1, 2)
plt.scatter(times, attitudes[:, 1],s=1)
plt.title('Angle 2 vs Temps')
plt.xlabel('Temps')
# plt.xlim(29,29.1)
plt.ylabel('Angle 2')

plt.subplot(3, 1, 3)
plt.scatter(times, attitudes[:, 2],s=1)
plt.title('Angle 3 vs Temps')
# plt.xlim(29,29.1)
plt.xlabel('Temps')
plt.ylabel('Angle 3')

plt.tight_layout()
plt.show()


# def windowed_kabsch_umeyama(A, B, w=500):
#     assert A.shape == B.shape
#     n, m = A.shape
#     rotations = []
#
#     for i in range(w, n-w):
#         window_A = A[i-w:i+w+1]
#         window_B = B[i-w:i+w+1]
#
#         EA = np.mean(window_A, axis=0)
#         EB = np.mean(window_B, axis=0)
#         VarA = np.mean(np.linalg.norm(window_A - EA, axis=1) ** 2)
#
#         H = ((window_A - EA).T @ (window_B - EB)) / (2*w+1)
#         U, D, VT = np.linalg.svd(H)
#         d = np.sign(np.linalg.det(U) * np.linalg.det(VT))
#         S = np.diag([1] * (m - 1) + [d])
#
#         R = U @ S @ VT
#         rotations.append(R)
#
#     return rotations




# def calculate_rotations(A):
#     """
#     A is a 3D array where A[t, i, j] gives the jth coordinate of the ith antenna at time t.
#     """
#     num_time_steps = A.shape[0]
#     print(num_time_steps)
#     rotations = []
#
#     for t in range(1, num_time_steps):
#         A_t = A[t, :, :]
#         A_t_1 = A[t-1, :, :]
#         R, _, _ = kabsch_umeyama(A_t, A_t_1)
#         rotations.append(R)
#
#     return rotations
#
# # Assemble the 3D matrix for all antennas
# num_time_steps = A1.shape[0]
# num_antennas = 4
# A = np.zeros((num_time_steps, num_antennas, 3))
# A[:, 0, :] = A1
# A[:, 1, :] = A2
# A[:, 2, :] = A3
# A[:, 3, :] = A4
#
# # Calculate rotations for all time steps
# R_totals = calculate_rotations(A)
# print(A4[1:])
# x_transduceur = A4[1:]  + R_totals @ np.array([6.40,0,16.53])


#
# R_roll = windowed_kabsch_umeyama(A1, A2)
# print(len(R_roll))
#
# R_pitch = windowed_kabsch_umeyama(A1, A4)
# print(len(R_pitch))
#
# R_yaw = windowed_kabsch_umeyama(A2, A4)
# print(len(R_yaw))
#
# # 1. Assurez-vous que les listes ont la même longueur
# assert len(R_yaw) == len(R_pitch) == len(R_roll), "Les listes de matrices de rotation ont des longueurs différentes."
#
# # 2. Calculez le produit de chaque ensemble de matrices de rotation pour chaque fenêtre
# R_totals = [yaw @ pitch @ roll for yaw, pitch, roll in zip(R_yaw, R_pitch, R_roll)]
#
# # 3. Calculez les angles d'Euler pour chaque matrice de rotation
# eulers = np.array([rotationMatrixToEulerAngles(R) for R in R_totals])
#
# x_transduceur = A4[500:-500]  + R_totals @ np.array([6.40,0,16.53])
#
# # Diviser les données en x, y et z
# data = {
#     'x': x_transduceur[:, 0],
#     'y': x_transduceur[:, 1],
#     'z': x_transduceur[:, 2]
# }
#
# # Enregistrer les données dans un fichier .mat
# savemat('data.mat', data)
#
# import numpy as np
# import matplotlib.pyplot as plt
#
# # Création des subplots
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
#
# # Plot de x contre y
# ax1.plot(A4[1:,0], A4[1:,1], label='A4', color='r')
# ax1.plot(x_transduceur[:,0], x_transduceur[:,1], label='x_transduceur', color='b')
# ax1.set_xlabel('X')
# ax1.set_ylabel('Y')
# ax1.legend()
# ax1.grid(True)
#
# # Plot de z en fonction du temps (l'index est utilisé comme temps)
# ax2.plot(A4[1:,2], label='A4', color='r')
# ax2.plot(x_transduceur[:,2], label='x_transduceur', color='b')
# ax2.set_xlabel('Time (index)')
# ax2.set_ylabel('Z')
# ax2.legend()
# ax2.grid(True)
#
# plt.tight_layout() # Pour un espacement approprié entre les subplots
# plt.show()
#
# # Convertir en degrés
# roulis = np.degrees(eulers[:, 0])
# tangage = np.degrees(eulers[:, 1])
# lacet = np.degrees(eulers[:, 2])
#
# days = data_unit1['days'].flatten() - 59015
# times = data_unit1['times'].flatten()
# datetimes = ((days * 24 * 3600) + times)/3600
#
# datetime = datetimes[mask]
# datetime = datetime[1:]
# # datetime = datetime[:-1]
# print(len(datetime))
# print(len(roulis))

# Initialisation de la figure
# fig, axs = plt.subplots(3, 1, figsize=(10, 8))
#
# # Roulis
# axs[0].scatter(datetime, roulis, s=2, label='Roll', color='r')
# axs[0].set_ylabel('Angle (en degrés)')
# axs[0].set_title('Roll')
# axs[0].set_xlim(26,40)
# # axs[0].set_ylim(-5,5)
# axs[0].legend()
#
# # Tangage
# axs[1].scatter(datetime, tangage, s=2, label='Pitch', color='g')
# axs[1].set_ylabel('Angle (degrees)')
# axs[1].set_title('Pitch')
# axs[1].set_xlim(26,40)
# # axs[1].set_ylim(-5,5)
# axs[1].legend()
#
# # Lacet
# axs[2].scatter(datetime, lacet, s=2, label='Yaw', color='b')
# axs[2].set_xlabel('Time')
# axs[2].set_ylabel('Angle (degrees)')
# axs[2].set_title('Yaw')
# axs[2].set_xlim(26,40)
# # axs[2].set_ylim(-5,5)
# axs[2].legend()
#
# # Ajustement de l'espacement entre les subplots
# plt.tight_layout()
# plt.show()
