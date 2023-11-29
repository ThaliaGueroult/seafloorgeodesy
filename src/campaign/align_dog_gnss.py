#!/usr/bin/python3
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import datetime
import matplotlib.dates as mdates

# Chargement des données
data_unit = sio.loadmat('../../data/SwiftNav_Data/Unit3-camp_bis.mat')
data_DOG = sio.loadmat('../../data/DOG/DOG1-camp.mat')

# Manipulation temporelle pour GNSS
days = data_unit['days'].flatten() - 59015
times = data_unit['times'].flatten()
datetimes = (days * 24 * 3600) + times
condition_gnss = (datetimes/3600 >= 25) & (datetimes/3600 <= 40.9)
time_GNSS = datetimes[condition_gnss]/3600
lat = data_unit['lat'].flatten()[condition_gnss]
lon = data_unit['lon'].flatten()[condition_gnss]
elev = data_unit['elev'].flatten()[condition_gnss]
x = data_unit['x'].flatten()[condition_gnss]
y = data_unit['y'].flatten()[condition_gnss]
z = data_unit['z'].flatten()[condition_gnss]

# Manipulation temporelle pour DOG
data_DOG = data_DOG["tags"].astype(float)
acoustic_DOG = np.unwrap(data_DOG[:,1]/1e9*2*np.pi)/(2*np.pi)
offset = 68056+65
time_DOG = (data_DOG[:,0]+ offset)/3600
condition_DOG = (time_DOG >= 25) & (time_DOG <= 37)
time_DOG = time_DOG[condition_DOG]
acoustic_DOG = acoustic_DOG[condition_DOG]

# Créer une matrice pour les données DOG avec des NaN par défaut
acoustic_DOG_aligned = np.full(time_GNSS.shape, np.nan)

# Pour chaque temps dans time_GNSS, vérifiez si nous avons une donnée DOG correspondante
for i, t in enumerate(time_GNSS):
    # Trouvez l'index de la valeur la plus proche dans time_DOG
    idx = np.argmin(np.abs(time_DOG - t))

    # Si le temps le plus proche est "suffisamment" proche, prenez cette valeur
    if abs(time_DOG[idx] - t) < 1e-3:  # 1e-3 est une tolérance, elle peut être ajustée
        acoustic_DOG_aligned[i] = acoustic_DOG[idx]

# Matrices alignées
align_unit1_GNSS = np.vstack((time_GNSS, lat, lon, elev, x, y, z)).T
align_unit1_DOG = np.vstack((time_GNSS, acoustic_DOG_aligned)).T

print(align_unit1_GNSS)
print(align_unit1_DOG)

# Calculer la distance basée sur le temps de parcours
speed = 1515  # vitesse en m/s
distance = acoustic_DOG_aligned * speed

base_time = datetime.datetime(2000, 1, 1)  # La date de base n'a pas d'importance puisque nous ne nous soucions que de l'heure
time_GNSS = [base_time + datetime.timedelta(hours=t) for t in time_GNSS]


# Création des graphiques
plt.figure(figsize=(15, 15))

# Tracer x par rapport au temps
plt.subplot(5, 1, 1)
plt.plot(time_GNSS, x, label='X')
plt.title('X vs Time')
plt.xlabel('Time (hours)')
plt.ylabel('X Value')
plt.grid(True)
plt.legend()

# Tracer y par rapport au temps
plt.subplot(5, 1, 2)
plt.plot(time_GNSS, y, label='Y')
plt.title('Y vs Time')
plt.xlabel('Time (hours)')
plt.ylabel('Y Value')
plt.grid(True)
plt.legend()

# Tracer z par rapport au temps
plt.subplot(5, 1, 3)
plt.plot(time_GNSS, z, label='Z')
plt.title('Z vs Time')
plt.xlabel('Time (hours)')
plt.ylabel('Z Value')
plt.grid(True)
plt.legend()

# Tracer les données DOG alignées
plt.subplot(5, 1, 4)
plt.plot(time_GNSS, acoustic_DOG_aligned, label='Acoustic DOG Aligned', color='r')
plt.title('Acoustic DOG Aligned vs Time')
plt.xlabel('Time (hours)')
plt.ylabel('Acoustic DOG Value')
plt.grid(True)
plt.legend()

# Tracer la distance
plt.subplot(5, 1, 5)
plt.plot(time_GNSS, distance, label='Distance', color='g')
plt.title('Distance vs Time')
plt.xlabel('Time (hours)')
plt.ylabel('Distance (m)')
plt.grid(True)
plt.legend()
# Ajouter de l'espace entre les sous-graphiques
plt.subplots_adjust(hspace=10)

plt.tight_layout()
plt.show()

# Création des graphiques
plt.figure(figsize=(15, 15))
# Tracer z par rapport au temps
plt.subplot(2, 1, 1)
plt.plot(time_GNSS, z, label='Z')
plt.title('Z vs Time')
plt.xlabel('Time (hours)')
plt.ylabel('Z Value')
plt.grid(True)
plt.legend()

# Tracer les données DOG alignées
plt.subplot(2,1, 2)
plt.plot(time_GNSS, acoustic_DOG_aligned, label='Acoustic DOG Aligned', color='r')
plt.title('Acoustic DOG Aligned vs Time')
plt.xlabel('Time (hours)')
plt.ylabel('Acoustic DOG Value')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
