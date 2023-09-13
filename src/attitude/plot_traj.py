import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt


def load_and_process_data(path):
    """Charge et traite les données d'une unité."""
    data = sio.loadmat(path)
    days = data['days'].flatten() - 59015
    times = data['times'].flatten()
    datetimes = ((days * 24 * 3600) + times) / 3600
    x, y, z = data['x'].flatten(), data['y'].flatten(), data['z'].flatten()
    return datetimes, x, y, z


# Chemins des fichiers
paths = [
    '../../data/SwiftNav_Data/Unit1-camp_bis.mat',
    '../../data/SwiftNav_Data/Unit2-camp_bis.mat',
    '../../data/SwiftNav_Data/Unit3-camp_bis.mat',
    '../../data/SwiftNav_Data/Unit4-camp_bis.mat'
]

all_datetimes, all_xs, all_ys, all_zs = [], [], [], []

# Chargez et traitez toutes les données
for path in paths:
    datetimes, x, y, z = load_and_process_data(path)
    all_datetimes.append(datetimes)
    all_xs.append(x)
    all_ys.append(y)
    all_zs.append(z)

# Trouver les temps communs
common_times = np.array(sorted(set(all_datetimes[0]).intersection(*all_datetimes[1:])))

# Filtrez les coordonnées en fonction des temps communs
coordinates_common = lambda data, datetimes: data[np.isin(datetimes, common_times)]
all_xs_common = [coordinates_common(x, dt) for x, dt in zip(all_xs, all_datetimes)]
all_ys_common = [coordinates_common(y, dt) for y, dt in zip(all_ys, all_datetimes)]
all_zs_common = [coordinates_common(z, dt) for z, dt in zip(all_zs, all_datetimes)]

# Visualisation des coordonnées x contre y pour chaque antenne
fig, ax = plt.subplots(figsize=(10, 10))

colors = ['r', 'purple', 'darkblue', 'g']
labels = ['Antenna 1', 'Antenna 2', 'Antenna 3', 'Antenna 4']

for i in range(4):
    ax.scatter(all_xs_common[i], all_ys_common[i], s=1, label=labels[i], color=colors[i])

ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.set_title('Y vs X for each Antenna')
ax.legend()

plt.tight_layout()
plt.show()


# Visualisez les données filtrées pour toutes les antennes
fig, ax = plt.subplots(3, 1, figsize=(10, 15))

colors = ['r', 'g', 'b', 'c']
labels = ['Antenna 1', 'Antenna 2', 'Antenna 3', 'Antenna 4']

for i in range(4):
    ax[0].scatter(common_times, all_xs_common[i], s=1, label=labels[i], color=colors[i])
    ax[1].scatter(common_times, all_ys_common[i], s=1, label=labels[i], color=colors[i])
    ax[2].scatter(common_times, all_zs_common[i], s=1, label=labels[i], color=colors[i])

titles = ['X Coordinate', 'Y Coordinate', 'Z Coordinate']
for i, title in enumerate(titles):
    ax[i].set_title(title)
    ax[i].legend()

plt.tight_layout()
plt.show()


# Calculez les différences pour chaque paire d'antennes
diff_xs = [all_xs_common[i] - all_xs_common[j] for i in range(3) for j in range(i+1, 4)]
diff_ys = [all_ys_common[i] - all_ys_common[j] for i in range(3) for j in range(i+1, 4)]
diff_zs = [all_zs_common[i] - all_zs_common[j] for i in range(3) for j in range(i+1, 4)]

# Créez les labels pour les paires
pair_labels = [f"Antenna {i+1} - Antenna {j+1}" for i in range(3) for j in range(i+1, 4)]

# Nouvelle liste de couleurs pour les différences
diff_colors = ['red', 'purple', 'darkblue', 'green', 'cyan', 'black']

# Visualisez les différences pour chaque paire d'antennes
fig, ax = plt.subplots(3, 1, figsize=(10, 15))

for i in range(6):  # Il y a 6 paires possibles avec 4 antennes
    ax[0].scatter(common_times, diff_xs[i], s=1, label=pair_labels[i], color=diff_colors[i])
    ax[1].scatter(common_times, diff_ys[i], s=1, label=pair_labels[i], color=diff_colors[i])
    ax[2].scatter(common_times, diff_zs[i], s=1, label=pair_labels[i], color=diff_colors[i])

titles = ['Difference in X Coordinate', 'Difference in Y Coordinate', 'Difference in Z Coordinate']
for i, title in enumerate(titles):
    ax[i].set_title(title)
    ax[i].legend()

plt.tight_layout()
plt.show()
