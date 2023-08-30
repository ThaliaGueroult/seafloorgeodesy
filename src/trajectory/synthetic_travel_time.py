# Function to generate a sinusoidal trajectory for the source
def generate_trajectory():
    """
    This function generates a sinusoidal trajectory for the source.
    Output:
        Returns a DataFrame containing the trajectory
    """
    # Define parameters for the sinusoidal trajectory
    lat_center = 31.45
    lon_center = 291.30
    elev_center = 5000
    amplitude = 1000
    period = 360

    # Generate the trajectory
    time = np.linspace(0, 360, 361)
    lat_traj = lat_center + amplitude * np.sin(np.radians(time))
    lon_traj = lon_center + amplitude * np.sin(np.radians(time))
    elev_traj = elev_center + amplitude * np.sin(np.radians(time))

    # Create a DataFrame to hold the trajectory
    trajectory = pd.DataFrame({'Time': time, 'Latitude': lat_traj, 'Longitude': lon_traj, 'Elevation': elev_traj})

    return trajectory
