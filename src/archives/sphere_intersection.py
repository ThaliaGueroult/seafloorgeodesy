from scipy.optimize import least_squares
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import pymap3d as pm
from scipy.io import savemat
from tqdm import tqdm
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import numpy as np
import random

def load_and_process_data(path):
    """Charge et traite les données d'une unité."""
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

print(filtered_data)
# Estimations initiales
levier_initial = [
    [16.6, -2.46, 15.24],
    [16.6, -7.39, 15.24],
    [6.40, -9.57, 15.24],
    [6.40, -2.46, 15.24]
]

def residual(point, antennas, distances):
    """Calcul du résidu pour l'optimisation des moindres carrés."""
    x, y, z = point
    residuals = []
    for (x_ant, y_ant, z_ant), d in zip(antennas, distances):
        distance_calculated = ((x - x_ant)**2 + (y - y_ant)**2 + (z - z_ant)**2)**0.5
        residuals.append(distance_calculated - d)
    return residuals


levier = {
    "antenne 1": np.array([16.53793082,   -2.24612876, 15.17258002]),
    "antenne 2": np.array([16.29752871,  -7.09213979, 14.93735477]),
    "antenne 3": np.array([5.60312943,  -8.84829343, 14.63103173]),
    "antenne 4": np.array([5.8376399,   -2.10820861, 15.22444008])
}

# Boucle sur chaque instant commun
transducer_positions = []
costs = []

for t_index, t in tqdm(enumerate(common_datetimes), total=len(common_datetimes), desc="Processing"):

    # Prenez les positions des antennes et les bras de levier à cet instant
    antennas_pos = [(data[1][t_index], data[2][t_index], data[3][t_index]) for data in filtered_data]
    distances = [np.linalg.norm(l) for l in levier.values()]
    # Estimation initiale de la position du transducteur (centre des antennes)
    x0 = np.mean(antennas_pos, axis=0)
    # Résolution des moindres carrés pour trouver la position du transducteur
    result = least_squares(residual, x0, args=(antennas_pos, distances))
    transducer_positions.append(result.x)
    costs.append(result.cost)


'''Dans le code commenté ci-dessous, je fais les résidus étendus
Cela permet d'essayer d'optimiser également les bras de leviers sachant que je ne sais pas exactement quels sont les bras de leviers
Je dispose uniquement des bras de leviers
'''
# def extended_residual(variables, antennas):
#     """Calcul du résidu pour l'optimisation des moindres carrés en incluant les bras de levier."""
#     x, y, z, *lever_arms = variables
#     lever_arms = np.array(lever_arms).reshape(4, 3)
#     residuals = []
#
#     for (x_ant, y_ant, z_ant), l in zip(antennas, lever_arms):
#         x_diff, y_diff, z_diff = x - x_ant, y - y_ant, z - z_ant
#         distance_real = np.sqrt(x_diff**2 + y_diff**2 + z_diff**2)
#         distance_ideal = np.linalg.norm(l)
#         residuals.append(distance_real - distance_ideal)
#     return residuals
#

#
# # Stockage de tous les bras de levier estimés
# all_estimated_leverages = []
#
#
# lower_bounds = [-np.inf, -np.inf, -np.inf]
# upper_bounds = [np.inf, np.inf, np.inf]
#
# for lev in levier_initial:
#     for dim in lev:
#         lower_bounds.append(dim - 1)  # -1 mètre
#         upper_bounds.append(dim + 1)  # +1 mètre
#
# # Continuation de la boucle principale
# for t_index, t in tqdm(enumerate(common_datetimes), total=len(common_datetimes), desc="Processing"):
#     antennas_pos = [(data[1][t_index], data[2][t_index], data[3][t_index]) for data in filtered_data]
#
#     # Initial guess (position du transducteur + bras de levier initial)
#     x0 = np.mean(antennas_pos, axis=0).tolist() + levier_initial[0] + levier_initial[1] + levier_initial[2] + levier_initial[3]
#
#     result = least_squares(extended_residual, x0, args=(antennas_pos,), bounds=(lower_bounds, upper_bounds))
#     transducer_positions.append(result.x[:3])
#     all_estimated_leverages.append(np.array(result.x[3:]).reshape(4, 3))
#     costs.append(result.cost)
#
# # Calcul de la moyenne des bras de levier estimés
# average_leverages = np.mean(all_estimated_leverages, axis=0)
# print("Bras de levier moyens estimés:\n", average_leverages)

plt.figure(figsize=(10, 6))
plt.scatter(common_datetimes, costs,s = 1)
plt.xlabel('Datetime')
plt.ylabel('Cost (Sum of squared residuals)')
plt.title('Residuals over time')
plt.xlim(26,40)
plt.ylim(-5,5)
plt.grid(True)
plt.tight_layout()
plt.show()


Q1 = np.percentile(costs, 25)
Q3 = np.percentile(costs, 75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
filtered_costs = [cost for cost in costs if lower_bound <= cost <= upper_bound]

plt.figure(figsize=(10, 6))
plt.hist(filtered_costs, bins=50)
# plt.xlabel('Residuals')
# plt.ylabel('Cost (Sum of squared residuals)')
plt.title('Histogram of the residuals')
plt.grid(True)
plt.tight_layout()
plt.show()

# 1. Enregistrement des positions du transducteur
transducer_positions = np.array(transducer_positions)  # Convertir en numpy array
x_transducer = np.column_stack((common_datetimes, transducer_positions))

# Plot 2D de x et y du transducteur avec z comme couleur
plt.figure(figsize=(10, 6))
sc = plt.scatter(x_transducer[:, 1], x_transducer[:, 2], c=x_transducer[:, 3], cmap='viridis')
plt.colorbar(sc, label='Profondeur (z)')

# Ajouter les x, y des antennes à ce plot
colors = ['red', 'blue', 'green', 'orange']
labels = ["antenne 1", "antenne 2", "antenne 3", "antenne 4"]

antennas_pos_first_timestamp = [(data[1][:], data[2][:], data[3][:]) for data in filtered_data]

for (x_ant, y_ant, _), color, label in zip(antennas_pos_first_timestamp, colors, labels):
    plt.scatter(x_ant, y_ant, color=color, label=label)


# Légende et titres
plt.legend()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Positions 2D du transducteur et des antennes')
plt.tight_layout()
plt.show()

# Supposons que transducer_positions soit une liste de tuples (x, y, z) que vous avez déjà calculés.
x = [pos[0] for pos in transducer_positions]
y = [pos[1] for pos in transducer_positions]
z = [pos[2] for pos in transducer_positions]
times = common_datetimes

# 1. Convertir les coordonnées ECEF en coordonnées géodésiques
lat, lon, elev = pm.ecef2geodetic(transducer_positions[:, 0],
                                  transducer_positions[:, 1],
                                  transducer_positions[:, 2],
                                  ell=None, deg=True)
print(elev)
# 2. Préparer le dictionnaire data_to_save
data_to_save = {
    'x': transducer_positions[:, 0],
    'y': transducer_positions[:, 1],
    'z': transducer_positions[:, 2],
    'times': common_datetimes,
    'lat': lat,
    'lon': lon,
    'elev': elev
}

# 3. Enregistrement dans matrix_transduceur.mat
sio.savemat('data_transduceur.mat', data_to_save)


def create_sphere(center, radius):
    """ Crée une sphère avec un rayon donné autour d'un point central. """
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
    return x, y, z

# Prendre 10 instants aléatoires
random_indices = random.sample(range(len(common_datetimes)), 3)

# ...
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import numpy as np
import random
# Pour chaque instant aléatoire
for index in random_indices:
    # Créez une figure avec Plotly
    fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scatter3d'}]])

    legend_added = set()  # pour suivre les labels déjà ajoutés à la légende

    # Points d'antenne pour cet instant
    antennas_pos = [(data[1][index], data[2][index], data[3][index]) for data in filtered_data]
    for i, position in enumerate(antennas_pos):
        # Dessinez la sphère avec un rayon égal à la distance au transducteur
        distance = np.linalg.norm(levier[labels[i]])
        x, y, z = create_sphere(position, distance)

        show_legend = labels[i] not in legend_added
        fig.add_trace(go.Surface(x=x, y=y, z=z, opacity=0.3, colorscale=[[0, colors[i]], [1, colors[i]]], showscale=False, showlegend=show_legend, name=labels[i]))

        if show_legend:  # Ajoutez une trace fictive juste pour la légende
            fig.add_trace(go.Scatter3d(x=[None], y=[None], z=[None], mode='markers', marker=dict(size=10, color=colors[i]), name=labels[i], showlegend=True))
            legend_added.add(labels[i])

        fig.add_trace(go.Scatter3d(x=[position[0]], y=[position[1]], z=[position[2]], mode='markers', marker=dict(size=10, color=colors[i]), name=labels[i], showlegend=False))

        # Trace la ligne entre l'antenne et le point du transducteur
        transducer_pos = transducer_positions[index]
        fig.add_trace(go.Scatter3d(x=[position[0], transducer_pos[0]], y=[position[1], transducer_pos[1]], z=[position[2], transducer_pos[2]], mode='lines', line=dict(color='grey', width=2), showlegend=False))

    # Trace des lignes entre les antennes a1-a2, a2-a3, a3-a4, a4-a1
    antenna_pairs = [(0, 1), (1, 2), (2, 3), (3, 0)]
    for pair in antenna_pairs:
        fig.add_trace(go.Scatter3d(x=[antennas_pos[pair[0]][0], antennas_pos[pair[1]][0]],
                                   y=[antennas_pos[pair[0]][1], antennas_pos[pair[1]][1]],
                                   z=[antennas_pos[pair[0]][2], antennas_pos[pair[1]][2]],
                                   mode='lines', line=dict(color='blue', width=2), showlegend=False))

    # Dessinez la position estimée du transducteur
    fig.add_trace(go.Scatter3d(x=[transducer_positions[index][0]], y=[transducer_positions[index][1]], z=[transducer_positions[index][2]], mode='markers', marker=dict(size=10, color='black', symbol='x'), name="Estimated Position"))

    fig.update_layout(title=f"Instant {index}", scene=dict(aspectmode="cube", camera=dict(up=dict(x=0, y=0, z=1), eye=dict(x=1.25, y=1.25, z=1.25))),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.show()

# Prendre un instant aléatoire
random_index = random.choice(range(len(common_datetimes) - 20))

# Créez une figure avec Plotly
fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scatter3d'}]])

# Points d'antenne et du transducteur pour les 20 instants qui suivent l'instant aléatoire
for offset in range(20):
    index = random_index + offset
    antennas_pos = [(data[1][index], data[2][index], data[3][index]) for data in filtered_data]

    for i, position in enumerate(antennas_pos):
        # Tracez la position de chaque antenne
        fig.add_trace(go.Scatter3d(x=[position[0]], y=[position[1]], z=[position[2]], mode='markers', marker=dict(size=10, color=colors[i]), name=f"Antenna {i+1} at t={offset}", showlegend=(offset==0)))

        # Tracer le mouvement des antennes avec des lignes
        if offset > 0:
            previous_position = [(data[1][index-1], data[2][index-1], data[3][index-1]) for data in filtered_data][i]
            fig.add_trace(go.Scatter3d(x=[previous_position[0], position[0]], y=[previous_position[1], position[1]], z=[previous_position[2], position[2]], mode='lines', line=dict(color=colors[i], width=2), showlegend=False))

    # Dessinez la position du transducteur pour cet instant
    transducer_pos = transducer_positions[index]
    fig.add_trace(go.Scatter3d(x=[transducer_pos[0]], y=[transducer_pos[1]], z=[transducer_pos[2]], mode='markers', marker=dict(size=10, color='black', symbol='x'), name=f"Transducer at t={offset}", showlegend=(offset==0)))

    # Tracer le mouvement du transducteur avec des lignes
    if offset > 0:
        previous_transducer_pos = transducer_positions[index-1]
        fig.add_trace(go.Scatter3d(x=[previous_transducer_pos[0], transducer_pos[0]], y=[previous_transducer_pos[1], transducer_pos[1]], z=[previous_transducer_pos[2], transducer_pos[2]], mode='lines', line=dict(color='black', width=2), showlegend=False))

# Paramètres d'affichage et affichage de la figure
fig.update_layout(title=f"Movement from instant {random_index} to {random_index + 19}", scene=dict(aspectmode="cube", camera=dict(up=dict(x=0, y=0, z=1), eye=dict(x=1.25, y=1.25, z=1.25))))
fig.show()
