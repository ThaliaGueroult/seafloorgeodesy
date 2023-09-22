from scipy.io import loadmat
import os
import matplotlib.pyplot as plt
import numpy as np

def load_matrix_from_file(dossier):
    """Load matrix from a given folder"""
    output_file_path = os.path.join(dossier, 'global_table_esv.mat')
    loaded_data = loadmat(output_file_path)
    return loaded_data

# Load matrices
dossier1 = 'castbermuda/global_table_interp'
dossier2 = 'GDEM/global_table_interp'

data1 = load_matrix_from_file(dossier1)
data2 = load_matrix_from_file(dossier2)

matrice1 = data1['matrice']
matrice2 = data2['matrice']

# Load angles and distances
angles = data1['angle'][0]
distances = data1['distance'][0]

# Compute the difference
difference = matrice1 - matrice2

# Compute statistics
mean_diff = np.mean(difference)
std_diff = np.std(difference)
min_diff = np.min(difference)
max_diff = np.max(difference)

fig, ax = plt.subplots(1, 2, figsize=(15, 6))

# Histogram on the first subplot
ax[0].hist(difference.ravel(), bins=50, edgecolor="k", alpha=0.7)
ax[0].set_title('Histogram of Differences between ESV tables')
ax[0].set_xlabel('Difference')
ax[0].set_ylabel('Frequency')
# Annotate with statistics
stats_text = (f"Mean: {mean_diff:.2f} m/s \n"
              f"Std Dev: {std_diff:.2f} m/s\n"
              f"Min: {min_diff:.2f} m/s\n"
              f"Max: {max_diff:.2f} m/s")
ax[0].annotate(stats_text, xy=(0.05, 0.95), xycoords='axes fraction',
               verticalalignment='top', horizontalalignment='left',
               bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="aliceblue"))
# Image difference on the second subplot
cax = ax[1].imshow(difference, cmap="Spectral_r", aspect='auto', extent=[angles.min(), angles.max(), distances.min(), distances.max()], origin='lower')
ax[1].set_title('Differences between ESV tables')
ax[1].set_xlabel('Angles (deg)')
ax[1].set_ylabel('Distances (m)')
fig.colorbar(cax, ax=ax[1], label='Difference (m/s)')

plt.tight_layout()
plt.show()
