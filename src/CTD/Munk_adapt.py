import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

def sound_velocity(z, z1, B, C1):
    '''This function computes the Munk profile of sound Velocity
    This is from DOI : 10.1121/1.1914492, Munk 1979
    z : depth (positive)
    z1 : axis with minimum velocity
    B : scale depth
    C1 :
    eps : perturbation coefficient
    eta :  dimensionless distance beneath axis
    '''
    eps = 1/2*B*0.0114e-3 #gamma_a=0.0114 km^-1 is the adiabatic velocity gradient
    eta = 2*(z-z1)/B
    return C1*(1.0+eps*(eta-1+np.exp(-eta)))

# Load Sound velocity profile to match
data_file2 = "../../data/AE2008_CTD/AE2008_PostDeployment_Cast2_deriv.cnv"
data2 = np.loadtxt(data_file2,skiprows=348)
depths_cast2 = data2[:,0]
depths_cast2 = depths_cast2[:np.argmax(depths_cast2)]
sv_delgrosso_cast2 = data2[:,21][:len(depths_cast2)]

def objective(x, z, c_real):
    z1, B1, C1 = x
    c_calculated = sound_velocity(z, z1, B1, C1)
    z1_calculated = z[np.argmin(c_calculated)]
    return ((c_calculated - c_real) ** 2).sum() + ((z1_calculated - z1) ** 2)

# Valeurs de départ pour les coefficients z1, B1 et C1
x0 = [1300, 1300, 1500]

# Minimisation de la fonction objectif
res = optimize.minimize(objective, x0, args=(depths_cast2, sv_delgrosso_cast2))

# Coefficients optimaux
z1_opt, B1_opt, C1_opt = res.x

# Célérité calculée avec les coefficients optimaux
c_opt = sound_velocity(depths_cast2, z1_opt, B1_opt, C1_opt)

# Affichage des résultats
print("Coefficients optimaux : z1 = {}, B1 = {}, C1 = {}".format(z1_opt, B1_opt, C1_opt))
print("Célérité réelle : {}".format(sv_delgrosso_cast2))
print("Célérité calculée avec les coefficients optimaux : {}".format(c_opt))


fig, ax = plt.subplots(figsize=(16, 10))
ax.plot(sv_delgrosso_cast2,depths_cast2,label='Bermuda, Princeton 2020')
ax.plot(c_opt, depths_cast2,label='Munk Model, 1979')
# ax.figtext("Coefficients optimaux : z1 = {}, B1 = {}, C1 = {}".format(z1_opt, B1_opt, C1_opt))
ax.set_xlabel('Sound Velocity (m/s)')
ax.set_ylabel('Depth (m)')
ax.legend()
plt.title('Comparison of Sound Velocity Profile')
plt.show()
