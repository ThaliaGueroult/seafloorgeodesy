import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

def compute_sound_velocity(z, z1, B, C1):
    """
    Compute the Munk profile of sound velocity based on DOI: 10.1121/1.1914492, Munk (1979).
    Parameters:
    z (float): Depth (positive).
    z1 (float): Axis with minimum velocity.
    B (float): Scale depth.
    C1 (float): Coefficient for the equation.

    Returns:
    float: Calculated sound velocity based on Munk profile.
    """
    eps = 1 / 2 * B * 0.0114e-3  # gamma_a=0.0114 km^-1 is the adiabatic velocity gradient
    eta = 2 * (z - z1) / B
    return C1 * (1.0 + eps * (eta - 1 + np.exp(-eta)))

# Load sound velocity profile data
data_file_path = "../../data/AE2008_CTD/AE2008_PostDeployment_Cast2_deriv.cnv"
data = np.loadtxt(data_file_path, skiprows=348)
depth_values = data[:, 0]
depth_values = depth_values[:np.argmax(depth_values)]
sv_delgrosso = data[:, 21][:len(depth_values)]

def objective_function(params, z, c_real):
    """
    Objective function to minimize the sum of squared differences
    between calculated and real sound velocity.
    """
    z1, B1, C1 = params
    c_calculated = compute_sound_velocity(z, z1, B1, C1)
    return ((c_calculated - c_real) ** 2).sum()

# Initial guesses for coefficients z1, B1, and C1
initial_guess = [1300, 1300, 1500]

# Minimize the objective function
result = optimize.minimize(objective_function, initial_guess, args=(depth_values, sv_delgrosso))

# Optimal coefficients
z1_optimal, B1_optimal, C1_optimal = result.x

# Sound velocity calculated with optimal coefficients
calculated_velocity = compute_sound_velocity(depth_values, z1_optimal, B1_optimal, C1_optimal)

# Print and plot results
print(f"Optimal coefficients: z1 = {z1_optimal}, B1 = {B1_optimal}, C1 = {C1_optimal}")
print(f"Actual sound velocity: {sv_delgrosso}")
print(f"Calculated sound velocity with optimal coefficients: {calculated_velocity}")

fig, ax = plt.subplots(figsize=(16, 10))
ax.plot(sv_delgrosso, depth_values, label='Bermuda, Princeton 2020')
ax.plot(calculated_velocity, depth_values, label='Munk Model, 1979')
ax.set_xlabel('Sound Velocity (m/s)')
ax.set_ylabel('Depth (m)')
ax.legend()
plt.title('Comparison of Sound Velocity Profiles')
plt.show()
