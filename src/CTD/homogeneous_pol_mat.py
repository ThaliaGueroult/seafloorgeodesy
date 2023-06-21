import numpy as np
import scipy.io
from read_GDEM import extract_gdem
from swpressure import *
import matplotlib.pyplot as plt

def define_coef_matrix():
    ''' this function defines the matrix of homogeneous polynome
    for Del Grosso Sound Velocity method (1974)
    https://doi.org/10.1121/1.1903388
    '''
    #define coef max
    deg_max_S = 3
    deg_max_T = 3
    deg_max_P = 3

    #create a zero array with size (deg_max_S + 1, deg_max_T + 1, deg_max_P + 1)
    coef = np.zeros((deg_max_S + 1, deg_max_T + 1, deg_max_P + 1))

    #fill coefficient value
    coef[0, 0, 0] = 1402.392
    coef[0, 0, 1] = 0.156059257041e0
    coef[0, 0, 2] = 0.244998688441e-4
    coef[0, 0, 3] = -0.88339233251e-8
    coef[0, 1, 0] = 0.501109398813e1
    coef[0, 1, 1] = 0.635191613389e-2
    coef[0, 1, 2] = -0.159349479045e-5
    coef[0, 1, 3] = 0.522116437235e-9
    coef[0, 2, 0] = -0.550946843172e-1
    coef[0, 2, 2] = 0.265484716608e-7
    coef[0, 3, 0] = 0.221535969240e-3
    coef[0, 3, 1] = -0.438031096213e-6
    coef[1, 0, 0] = 0.132952290781e1
    coef[1, 1, 0] = -0.127562783426e-1
    coef[1, 1, 1] = -0.340597039004e-3
    coef[1, 2, 0] = 0.968403156410e-4
    coef[2, 0, 0] = 0.128955756844e-3
    coef[2, 0, 2] = -0.161674495909e-8
    coef[2, 1, 1] = 0.485639620015e-5

    return coef

def c_delgrosso(S, T, P, lat = 45):
    '''computes sound velocity using
    Del Grosso (1974) and the homogenous polynome coefficient
    inputs :
        S in PSU
        T in Degree Celsius
        P in kg/cm^2.  To get to this from dbars we  must divide by "g".
        From the UNESCO algorithms (referring to ANON (1970) BULLETIN GEODESIQUE)
        we have this formula for g as a function of latitude and pressure.
        We set latitude to 45 degrees for convenience!
    outputs :
        c in m/s, the sound velocity or sound velocity profile
    '''
    # XX = np.sin(lat * np.pi / 180)
    # GR = 9.780318 * (1.0 + (5.2788e-3 + 2.36e-5 * XX) * XX) + 1.092e-6 * P
    # P = P / GR
    P = P * 1.01972/10

    coef = define_coef_matrix()

    c = 0
    for deg_S in range(coef.shape[0]):
        for deg_T in range(coef.shape[1]):
            for deg_P in range(coef.shape[2]):
                c += coef[deg_S, deg_T, deg_P] * (S ** deg_S) * (T ** deg_T) * (P ** deg_P)

    return c

def std_c_delgrosso(S, T, P, std_s, std_t, std_p, lat = 45):
    '''
    Computes sound velocity standard deviation
    using Del Grosso formula
    Formula made manually with rules of error propagation
    inputs :
        S, std_S in PSU
        T, std_T in Degre Celsius
        P, std_P in dBar
    outputs :
        std_c in m/s, standard deviation of the sound velocity
    # '''
    P = P * 1.01972/10
    std_p = std_p * 1.01972/10


    coef = define_coef_matrix()
    std_c_2 = 0
    for deg_S in range(coef.shape[0]):
        for deg_T in range(coef.shape[1]):
            for deg_P in range(coef.shape[2]):
                std_c_2 += (coef[deg_S, deg_T, deg_P] * std_s * deg_S * (S ** (deg_S - 1)) * (T ** deg_T) * (P ** deg_P)) ** 2
                std_c_2 += (coef[deg_S, deg_T, deg_P] * std_t * (S ** deg_S) * deg_T * (T ** (deg_T - 1)) * (P ** deg_P)) ** 2
                std_c_2 += (coef[deg_S, deg_T, deg_P] * std_p * (S ** deg_S) * (T ** deg_T) * deg_P * (P ** (deg_P - 1))) ** 2
    return np.sqrt(std_c_2)


if __name__ == '__main__':
    map = scipy.io.loadmat('../../data/GDEM/geomap_GDEMV.mat')
    data = scipy.io.loadmat('../../data/GDEM/Jun_GDEMV.mat')
    lat = 31.44
    lon = -68.693

    temp, sal, std_temp, std_sal, depth = extract_gdem(lat, lon, map, data)
    depth=map['depth'][:,0]
    std_p=np.zeros_like(std_temp)
    #convert depth into pressure in dB and compute soundspeed with Del Grosso equation
    pressure=swpressure(depth,lat)
    sv=c_delgrosso(sal,temp,pressure)
    std_c=std_c_delgrosso(sal, temp, pressure, std_sal, std_temp,std_p)
    print(std_c)

    lower = sv - std_c
    upper = sv + std_c
    plt.fill_betweenx(depth, lower, upper, facecolor='orange', alpha=0.3)

    plt.plot(sv,depth)
    # plt.errorbar(sv, depth, xerr=std_c, linestyle='None')
    plt.gca().invert_yaxis()
    plt.show()

    print(c_delgrosso(39,26,145))
