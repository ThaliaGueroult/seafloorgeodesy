import numpy as np
from scipy.interpolate import interp1d

'''C'est une traduction chatgpt de gps2rng et de dog2rng du code Matlab de Tjschuh
dans GPSTOOLS, que je n'ai pas encore revue'''

def gps2rng(files, meth='ave', xyz=np.array([2e6, -4.5e6, 3e6]), v=1500, depth=5225):
    # Function to calculate the slant range to a target using GPS data from multiple units

    defval = lambda arg, default: arg if arg is not None else default

    # How the possibly multiple receivers will be jointly considered
    meth = defval(meth, 'ave')

    # Combine all datasets into 1 large matrix
    # Assuming files contain GPS data, modify this part if it's different
    dxyz = np.mean(np.stack([np.load(file)['xyz'] for file in files]), axis=0)

    # Default beacon location is DOG 1 drop off location from June 2020 cruise
    xyz = defval(xyz, np.array([1.977967, -5.073198, 3.3101016]) * 1e6)

    # Put in water depth read off the GPSTRAJECT map
    depth = defval(depth, 5225)

    # This is the forward model
    # Calculate slant range between ship and beacon for each second in meters
    sr = np.sqrt(np.sum((dxyz - xyz)**2, axis=1))

    # Calculate slant time from slant range and v [s]
    st = sr / v

    # Optional output
    return st, dxyz, sr, xyz, v

def retimei(tags, n):
    # Function to correct NaNs and add extra NaNs when more than n consecutive missed detections occur
    difft = np.diff(tags[:, 1])
    missing_idx = np.where(difft > n)[0]
    for idx in missing_idx:
        num_missing = int(np.floor(difft[idx] / n))
        tags = np.insert(tags, idx + 1, np.array([tags[idx, 0]] * num_missing), axis=0)
        tags[idx + 1 : idx + 1 + num_missing, 1] += np.arange(1, num_missing + 1) * n
    return tags

def dog2rng(filenames):
    # Function to convert DOG data into range measurements

    if len(filenames) != 4:
        raise ValueError("Must have exactly 4 sets of DOG data")

    # DOG 1-4 approx positions
    xyz = np.array([[1.977967, -5.073198, 3.3101016],
                    [1.9765409, -5.074798706, 3.308558817],
                    [1.980083296, -5.073417583, 3.308558817],
                    [1.978732384, -5.075186123, 3.30666684]]) * 1e6

    # Threshold for NaN replacements
    n = np.array([1, 1, 1, 1])

    # St offset to be used during plotting so GNSS and acoustic data line up
    stoff = np.array([-100, 0, 900, 8800])

    # Create a figure and subplots
    fig, axes = plt.subplots(4, 1, figsize=(8, 10), sharex=True)

    # Loop through each DOG file
    for i, filename in enumerate(filenames):
        # Generate GNSS slant times using DOG approx position and gps2rng function
        v = 1500
        depth = 5225
        st, dxyz, sr, _, _ = gps2rng(['Unit1-camp.npy', 'Unit2-camp.npy', 'Unit3-camp.npy', 'Unit4-camp.npy'],
                                     xyz=xyz[i], v=v, depth=depth)

        # Load the dog tags
        tags = np.load(filename)

        # Do the alignment of the acoustic data with the GNSS data

        # Always want to unwrap before anything else
        tags[:, 1] = np.unwrap(tags[:, 1] / 1e9 * 2 * np.pi) / (2 * np.pi)

        # Eyeball an offset and cut it off
        tags = tags[ofx1[i] :, 0:2] + ofy1[i]

        # Now need to stick in extra NaNs
        tags = retimei(tags, n[i])

        # Speed corrections
        tags[ofxy1[i] + 1 :, 1] += ofy2[i]
        tags[ofxy2[i] : ofxy3[i], 1] += ofy3[i]

        # Eyeball a new offset and cut that off
        tags = tags[ofx2[i] :, 0:2]

        # Add in st offset for plot alignment
        tags[:, 1] -= stoff[i]

        # If we have non-NaN ofx3 value, build that in
        if not np.isnan(ofx3[i]):
            tags = tags[: ofx3[i], 0:2]

        # Make data same length as st
        if len(tags) > len(st):
            tags = tags[:len(st)]
        elif len(tags) < len(st):
            tags = np.vstack((tags, np.full((len(st) - len(tags), 2), np.nan)))

        # Plot GNSS slant times from ship positions
        axes[i].plot(st, 'b')

        # Plot acoustic slant times
        axes[i].plot(tags[:, 0], tags[:, 1], 'r.', markersize=3)

        # Set plot properties
        axes[i].set_xlim([0, 3600 * 24])
        axes[i].set_ylim([0.85 * np.min(st), 1.05 * np.max(st)])
        axes[i].set_xticks(np.arange(0, 3600 * 24, 3600 * 3))
        axes[i].set_xticklabels([])
        if i == 3:
            axes[i].set_xlabel('time [h]')
            axes[i].set_xticklabels(np.arange(0, 24, 3))
        axes[i].grid(True)
        axes[i].set_title(f'C-DOG approx position: x = {xyz[i, 0] * 1e-3:.4f} km, y = {xyz[i, 1] * 1e-3:.4f} km, z = {xyz[i, 2] * 1e-3:.4f} km', fontsize=8)

    # Add ylabel
    fig.text(0.04, 0.5, 'slant range time [s]', va='center', rotation='vertical')

    # Plot legend
    axes[0].plot([], [], 'b-', label='GNSS')
    axes[0].plot([], [], 'r-', label='Acoustic')
    axes[0].legend(loc='northeast', fontsize=4)

    # Save figure as PDF
    plt.savefig('dog2rng-plot.pdf')

    # Optional output
    return st, xtags, ytags
