import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

data = sio.loadmat('../../data/DOG/DOG1/DOG1-camp.mat')
print(data)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def dog2rng(fname):
    # if not 4 DOGS used in input, don't run the code
    if len(fname) != 4:
        raise ValueError('Must have exactly 4 sets of DOG data')

    # DOG 1-4 approx positions
    xyz = np.array([[1.977967, -5.073198, 3.3101016],
                    [1.9765409, -5.074798706, 3.308558817],
                    [1.980083296, -5.073417583, 3.308558817],
                    [1.978732384, -5.075186123, 3.30666684]]) * 1e6

    # Define other parameters (to-do: add more if needed)
    ofx1 = np.array([5823, 1, 1, 2250])
    ofx2 = np.array([4800, 12500, 12000, 1])
    ofx3 = np.array([np.nan, 86000, np.nan, 77500])
    ofy1 = np.array([0, 1, 1, 2])
    ofy2 = np.array([-4, -4, -3, -3])
    ofy3 = np.array([0, 0, 0, -1])
    ofxy1 = np.array([72856, 80678, 79625, 59490])
    ofxy2 = np.array([1, 1, 1, 22000])
    ofxy3 = np.array([1, 1, 1, 41000])
    n = np.array([1, 1, 1, 1])
    stoff = np.array([-100, 0, 900, 8800])
    stoff2 = np.array([17100, 17200, 18500, 17100])

    # Create a figure
    fig, axs = plt.subplots(4, 1, figsize=(8, 10), sharex=True)

    for i in range(4):
        # Generate GNSS slant times (st) using DOG approx position and GPS2RNG
        v = 1500
        depth = 5225
        st = gps2rng(['Unit1-camp.mat', 'Unit2-camp.mat', 'Unit3-camp.mat', 'Unit4-camp.mat'], 'ave', xyz[i], v, depth)

        # Plot GNSS slant times from ship positions
        axs[i].plot(st, 'b')
        axs[i].set_ylim(0.85 * np.min(st), 1.05 * np.max(st))
        axs[i].set_xlim(0, 3600 * 24)
        axs[i].grid(True)
        axs[i].set_xticks(np.arange(0, 3600 * 24, 3 * 3600))
        axs[i].set_xticklabels([])

        # Load the dog tags
        data = pd.read_csv(fname[i])  # Assuming data is stored in CSV format

        # Perform data alignment and processing (to-do: complete this part)

        # Save aligned data to a common matrix
        # Make data same length as st
        if len(data) > len(st):
            data = data.iloc[:len(st)]
        elif len(data) < len(st):
            data = pd.concat([data, pd.DataFrame({'Column1': [np.nan] * (len(st) - len(data))})])
        # Rename columns as per MATLAB code
        data.columns = ['Column1', 'Column2']
        xtags = data['Column1'].to_numpy()
        ytags = data['Column2'].to_numpy()

        # Other processing steps (to-do: complete this part)

        # Plot legend
        axs[i].plot([], [], 'g.', markersize=3, label='Acoustic')
        axs[i].legend(fontsize=4, loc='upper right')

        # Cosmetics
        axs[i].set_title(f'C-DOG approx position: x = {xyz[i, 0] * 1e-3:.3f} km, y = {xyz[i, 1] * 1e-3:.3f} km, z = {xyz[i, 2] * 1e-3:.3f} km', fontsize=8)
        txt = f'offset at {ofst[i]:.2f} hrs'
        axs[i].text(900, 0.9 * axs[i].get_ylim()[1], txt, fontsize=7)

    # Single ylabel workaround
    axs[0].set_ylabel('slant range time [s]')
    plt.xlabel('time [h]')
    plt.xticks(np.arange(0, 3600 * 24, 3 * 3600), [f'{i:d}' for i in range(0, 25, 3)])

    plt.tight_layout()
    plt.show()
    # Optional output
    return st, xtags, ytags

# Function gps2rng is not defined in the given code.
# You'll need to implement this function or use a suitable alternative.

# Sample usage
dog2rng(['DOG1-camp.csv', 'DOG2-camp.csv', 'DOG3-camp.csv', 'DOG4-camp.csv'])

import numpy as np
import matplotlib.pyplot as plt

def gps2rng(files, meth='ave', xyz=np.array([1.977967, -5.073198, 3.3101016]) * 1e6, v=1500, depth=5225):
    # Get extension from the first filename
    fname = files[0].split('.')[0]
    filex = fname.split('-')

    # If more than one input, list them all
    if len(files) > 1:
        if meth == 'ave':
            # More or less one line version of older MAT2COM.m
            dxyz = np.nanmean(np.reshape(np.concatenate([d['xyz'] for d in files]), (files[0]['xyz'].shape[0], len(files), 3)), axis=1)
    else:
        # Save yourself the trouble
        dxyz = files[0]['xyz']

    # Default beacon location is DOG 1 drop off location from June 2020 cruise
    if xyz is None:
        xyz = np.array([1.977967, -5.073198, 3.3101016]) * 1e6

    # Put in water depth read off the GPSTRAJECT map
    if depth is None:
        depth = 5225

    # Calculate slant range between ship and beacon for each second in meters
    sr = np.sqrt(np.sum((dxyz - xyz)**2, axis=1))
    # Calculate slant time from slant range and v [s]
    st = sr / v

    # Optional output
    varns = (st, dxyz, sr, xyz, v)
    return varns

# Function gebco is not defined in the given code.
# You'll need to implement this function or use a suitable alternative.

# Sample usage
files = [{'xyz': np.random.rand(10, 3) * 1e6} for _ in range(4)]
st, dxyz, sr, xyz, v = gps2rng(files)
print(st)
print(dxyz)
print(sr)
print(xyz)
print(v)
