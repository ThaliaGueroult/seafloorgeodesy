import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

data_DOG1 = sio.loadmat('../../data/DOG/DOG1-camp.mat')
data_DOG2 = sio.loadmat('../../data/DOG/DOG2-camp.mat')
data_DOG3 = sio.loadmat('../../data/DOG/DOG3-camp.mat')
data_DOG4 = sio.loadmat('../../data/DOG/DOG4-camp.mat')
print(data_DOG1)

tags_DOG1 = data_DOG1['tags']
tags_DOG2 = data_DOG2['tags']
tags_DOG3 = data_DOG3['tags']
tags_DOG4 = data_DOG4['tags']

# Transformation de la deuxième colonne
def transform(data):
    return np.unwrap(data[:, 1] / 1e9 * 2 * np.pi) / (2 * np.pi)

tags_DOG1_transformed = transform(data_DOG1['tags'])
tags_DOG2_transformed = transform(data_DOG2['tags'])
tags_DOG3_transformed = transform(data_DOG3['tags'])
tags_DOG4_transformed = transform(data_DOG4['tags'])


# Liste de couleurs pour chaque DOG et une couleur pour la ligne verticale
colors = ['r', 'g', 'b', 'y']
vline_color = 'cyan'  # 5ème couleur pour la ligne verticale

# Création de la figure et des subplots
fig, axs = plt.subplots(2, 2, figsize=(10,10))

# Ajout des données transformées et de la ligne verticale pour chaque DOG
def plot_data(ax, tags, tags_transformed, color):
    ax.scatter(tags[:,0], tags_transformed, color=color, s=5)
    ax.axvline(x=23200, color=vline_color, linestyle='--')
    ax.axvline(x=78000, color=vline_color, linestyle='--')
    ax.set_xlabel('PPS count (s)')
    ax.set_ylabel('Slant Range Time (s)')

plot_data(axs[0, 0], tags_DOG1, tags_DOG1_transformed, colors[0])
axs[0, 0].set_title('DOG1')

plot_data(axs[0, 1], tags_DOG2, tags_DOG2_transformed, colors[1])
axs[0, 1].set_title('DOG2')

plot_data(axs[1, 0], tags_DOG3, tags_DOG3_transformed, colors[2])
axs[1, 0].set_title('DOG3')

plot_data(axs[1, 1], tags_DOG4, tags_DOG4_transformed, colors[3])
axs[1, 1].set_title('DOG4')

# Ajustement pour éviter le chevauchement
plt.tight_layout()
plt.show()
