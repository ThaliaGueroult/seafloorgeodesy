import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

data = sio.loadmat('../../data/SwiftNav_Data/rawdata/Unit1-camp.mat')

def filter_outliers(lat, lon, elev, time):
    '''
    Filtre les valeurs aberrantes des données de latitude, longitude et élévation.

    Args:
    lat (array): Données de latitude.
    lon (array): Données de longitude.
    elev (array): Données d'élévation.

    Returns:
    lat_filt (array): Données de latitude filtrées.
    lon_filt (array): Données de longitude filtrées.
    elev_filt (array): Données d'élévation filtrées.
    '''
    # Calculer Q1 et Q3
    Q1 = np.percentile(elev, 25)
    Q3 = np.percentile(elev, 75)

    # Calculer l'IQR
    IQR = Q3 - Q1

    # Définir les seuils pour filtrer les valeurs aberrantes
    lower_threshold = Q1 - 1.5 * IQR
    upper_threshold = Q3 + 1.5 * IQR

    # Filtrer les valeurs aberrantes
    elev_filt = elev[(elev >= lower_threshold) & (elev <= upper_threshold)]
    # Filtrer les valeurs correspondantes dans lat et lon
    lat_filt = lat[(elev >= lower_threshold) & (elev <= upper_threshold)]
    lon_filt = lon[(elev >= lower_threshold) & (elev <= upper_threshold)]
    time_filt = time[(elev >= lower_threshold) & (elev <= upper_threshold)]

    trajectory = []

    for i in range (len(elev_filt)):
        point = (lat_filt[i], lon_filt[i], elev_filt[i])
        trajectory.append(point)

    return trajectory, time_filt

# Extraire les données
lat = data['d'][0]['lat'][0].flatten()
lon = data['d'][0]['lon'][0].flatten()
elev = data['d'][0]['height'][0].flatten()  # Flatten the elev numpy array

plt.hist(elev, bins = 100)
plt.show()


# Calculer Q1 et Q3
Q1 = np.percentile(elev, 25)
Q3 = np.percentile(elev, 75)

# Calculer l'IQR
IQR = Q3 - Q1

# Définir les seuils pour filtrer les valeurs aberrantes
lower_threshold = Q1 - 1.5 * IQR
upper_threshold = Q3 + 1.5 * IQR

# Filtrer les valeurs aberrantes
elev_filt = elev[(elev >= lower_threshold) & (elev <= upper_threshold)]
# Filtrer les valeurs correspondantes dans lat et lon
lat_filt = lat[(elev >= lower_threshold) & (elev <= upper_threshold)]
lon_filt = lon[(elev >= lower_threshold) & (elev <= upper_threshold)]

# Tracer les données avec une colormap pour elev_filt
plt.figure(figsize=(10, 6))
plt.scatter(lon_filt, lat_filt, c=elev_filt, cmap='viridis', marker='o', s=2)
plt.colorbar(label='Elevation (m)')
plt.xlabel('Longitude (deg)')
plt.ylabel('Latitude (deg)')
plt.title('Latitude vs. Longitude with Elevation Colormap (Filtered)')
plt.grid(True)

# plt.show()
