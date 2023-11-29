import scipy.io
import numpy as np
from read_GDEM import extract_gdem
from swpressure import *
import matplotlib.pyplot as plt

def soundspeed_delgrosso(S,T,D):
        '''
        This function compute sound speed profile from Delgrosso equation
        Del grosso uses pressure in kg/cm^2.  To get to this from dbars
        we  must divide by "g".  From the UNESCO algorithms (referring to
        ANON (1970) BULLETIN GEODESIQUE) we have this formula for g as a
        function of latitude and pressure.  We set latitude to 45 degrees
        for convenience!
        This formula has been taken from SBE Data Processing Software
        '''
        XX = np.sin(45 * np.pi / 180)
        GR = 9.780318 * (1.0 + (5.2788e-3 + 2.36e-5 * XX) * XX) + 1.092e-6 * D
        P = D / GR
        # This is from VSOUND.f.
        C000 = 1402.392
        DCT = (0.501109398873e1 - (0.550946843172e-1 - 0.221535969240e-3 * T) * T) * T
        DCS = (0.132952290781e1 + 0.128955756844e-3 * S) * S
        DCP = (0.156059257041e0 + (0.244998688441e-4 - 0.883392332513e-8 * P) * P) * P
        DCSTP = ((-0.127562783426e-1 * T * S + 0.635191613389e-2 * T * P + 0.265484716608e-7 * T * T * P * P
                - 0.159349479045e-5 * T * P * P
                + 0.522116437235e-9 * T * P * P * P
                - 0.438031096213e-6 * T * T * T * P)
            - 0.161674495909e-8 * S * S * P * P
            + 0.968403156410e-4 * T * T * S
            + 0.485639620015e-5 * T * S * S * P
            - 0.340597039004e-3 * T * S * P)
        ssp = C000 + DCT + DCS + DCP + DCSTP
        return ssp

def std_delgrosso(S, T, P, stdS, stdT, stdP):
    P = P * 1.01972 / 10
    stdP = stdP * 1.01972 / 10
    C000 = 1402.392

    dDCP_dP = 3 * 0.156059257041 * P**2 + 2 * (0.244998688441 * 10**-4 - 0.883392332513 * 10**-8 * P) * P - 0.438031096213 * 10**-6 * T**3
    dDCSTP_dP = 0.635191613389 * 10**-2 * T + 2 * 0.265484716608 * 10**-7 * T**2 * P**2 - 2 * 0.159349479045 * 10**-5 * T * P**2 + 3 * 0.522116437235 * 10**-9 * T * P**3 - 3 * 0.438031096213 * 10**-6 * T**2 * P - 2 * 0.161674495909 * 10**-8 * S**2 * P**2 + 2 * 0.968403156410 * 10**-4 * T**2 * S + 0.485639620015 * 10**-5 * T * S**2 * P - 0.340597039004 * 10**-3 * T * S
    dDCT_dT = 3 * (0.501109398873 * 10 - (0.550946843172 * 10**-1 - 0.221535969240 * 10**-3 * T) * T) * T**2
    dDCSTP_dT = -0.127562783426 * 10**-1 * S + 0.635191613389 * 10**-2 * P + 2 * 0.265484716608 * 10**-7 * T * P**2 - 2 * 0.159349479045 * 10**-5 * T * P**2 + 3 * 0.522116437235 * 10**-9 * T * P**3 - 3 * 0.438031096213 * 10**-6 * T**2 * P - 2 * 0.161674495909 * 10**-8 * S**2 * P**2 + 2 * 0.968403156410 * 10**-4 * T**2 * S + 2 * 0.485639620015 * 10**-5 * S**2 * P - 0.340597039004 * 10**-3 * S * P
    dDCS_dS = 2 * 0.132952290781 * S + 2 * 0.128955756844 * 10**-3
    dDCSTP_dS = -0.127562783426 * 10**-1 * T - 2 * 0.161674495909 * 10**-8 * S * P**2 + 0.968403156410 * 10**-4 * T**2 + 2 * 0.485639620015 * 10**-5 * T * S * P - 0.340597039004 * 10**-3 * T * P

    # Error propagation
    sigma_c = np.sqrt((dDCP_dP * stdP)**2 + (dDCT_dT * stdT)**2 + (dDCS_dS * stdS)**2)

    return (sigma_c)[0,:]



if __name__ == "__main__":
    map_data = scipy.io.loadmat('../../data/GDEM/geomap_GDEMV.mat')
    data = scipy.io.loadmat('../../data/GDEM/Jun_GDEMV.mat')
    lat = 31.44
    lon = -68.693

    temp, sal, std_temp, std_sal = extract_gdem(lat, lon, map_data, data)
    depth = map_data['depth']

    # Convert depth into pressure in dB and compute sound speed with Del Grosso equation
    pressure = swpressure(depth, lat)
    sv = soundspeed_delgrosso(sal, temp, pressure[:, 0])
    std_sv = std_delgrosso(sal, temp, pressure, std_sal, std_temp, 0)

    # Plotting the results
    plt.plot(sv, depth)
    plt.errorbar(sv, depth, xerr=std_sv, fmt='o', color='red', ecolor='gray', capsize=3)
    plt.xlabel('Sound Speed (m/s)')
    plt.ylabel('Depth')
    plt.title('Sound Speed Profile')
    plt.grid()
    plt.show()
