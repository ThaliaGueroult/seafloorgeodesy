import math
import matplotlib.pyplot as plt

def generate_trajectory():
    lat_min = 31.35
    lat_max = 31.55
    lon_min = 291.20
    lon_max = 291.40

    lat_range = lat_max - lat_min
    lon_range = lon_max - lon_min

    num_points = 10000  # Nombre de points sur la trajectoire
    angle_factor = 20  # Facteur d'échelle pour l'angle

    trajectory = []

    for i in range(num_points):
        t = float(i) / (num_points - 1)  # Valeur normalisée entre 0 et 1
        angle = t * 2 * math.pi * angle_factor
        x = math.sin(angle)
        y = math.sin(angle * 2) / 2

        lat = lat_min + (x + 1) * (lat_range / 2)
        lon = lon_min + (y + 1) * (lon_range / 2)
        elev = 5 * math.sin(angle/(10*(t+1)))  # Altitude sinusoidale entre -5 m et 5 m

        point = (lat, lon, elev)
        trajectory.append(point)

    return trajectory

# Génération de la trajectoire
trajectory = generate_trajectory()

# Extraction des coordonnées lat/lon pour le tracé
lats = [point[0] for point in trajectory]
lons = [point[1] for point in trajectory]
elevs = [point[2] for point in trajectory]

# Tracé de la trajectoire avec colormap pour la hauteur
plt.scatter(lons, lats, c=elevs, cmap='jet', s=5)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Trajectoire en forme de 8 avec colormap pour la hauteur (période plus courte)')
plt.colorbar(label='Altitude')
plt.grid(True)
plt.show()
