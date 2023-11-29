import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, BoundaryNorm

def read_ascii(file_path):
    data = np.loadtxt(file_path, skiprows=6)
    with open(file_path, 'r') as file:
        ncols = int(file.readline().split()[1])
        nrows = int(file.readline().split()[1])
        xllcorner = float(file.readline().split()[1])
        yllcorner = float(file.readline().split()[1])
        cellsize = float(file.readline().split()[1])

    lons = xllcorner + cellsize * np.arange(ncols)
    lats = yllcorner + cellsize * np.arange(nrows)

    return lons, lats, data

def plot_gebco(lons, lats, data, target_lon=None, target_lat=None):
    fig, ax = plt.subplots(figsize=(10, 10))

    # Déterminer les limites pour le colormap
    vmax = max(abs(data.min()), abs(data.max()))

    pc = ax.pcolormesh(lons, lats, data, shading='auto', cmap='terrain')

    if target_lon and target_lat:
        ax.scatter(target_lon, target_lat, color='red', marker='o')
        elevation = get_elevation_from_ascii(file_path, target_lon, target_lat)
        ax.text(target_lon, target_lat, f' {elevation:.2f}m', color='red', verticalalignment='bottom')

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("GEBCO Grid")
    ax.set_aspect('equal')

    # Créer la barre de couleurs avec la hauteur fixée
    pos = ax.get_position()
    cbar_ax = fig.add_axes([pos.x1 + 0.02, pos.y0, 0.03, pos.height])  # x, y, width, height
    fig.colorbar(pc, cax=cbar_ax, label="Elevation (m)")

    plt.show()


# Fonction précédente pour obtenir l'élévation à partir des coordonnées
def get_elevation_from_ascii(file_path, target_lon, target_lat):
    with open(file_path, 'r') as file:
        ncols = int(file.readline().split()[1])
        nrows = int(file.readline().split()[1])
        xllcorner = float(file.readline().split()[1])
        yllcorner = float(file.readline().split()[1])
        cellsize = float(file.readline().split()[1])
        nodata_value = float(file.readline().split()[1])
        col = int((target_lon - xllcorner) / cellsize)
        row = int((target_lat - yllcorner) / cellsize)
        if col < 0 or col >= ncols or row < 0 or row >= nrows:
            return None
        for _ in range(row):
            file.readline()
        line = file.readline()
        elevation = float(line.split()[col])
        if elevation == nodata_value:
            return None
        return elevation
lat = 31.46358041
lon = -68.70136571
file_path = "gebco_bermuda.asc"
lons, lats, data = read_ascii(file_path)
plot_gebco(lons, lats, data, lon, lat)
