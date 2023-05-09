import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from swpressure import *
from coordfromHeader import *


def lon_360to180(lon):
    '''
    This function allows to convert longitude from 0 to 360°
    to -180 to 180°
    '''
    lon=((lon - 180) % 360) - 180
    return(lon)

def plot_sal_temp(sal,temp,depth):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    # Tracer la courbe de température
    ax1.plot(temp, depth, 'b',label='Température',)
    ax1.set_xlabel('Température (°C)')
    ax1.set_ylabel('Profondeur (m)')
    ax1.legend()

    # Tracer la courbe de salinité
    ax2.plot(sal, depth,'g', label='Salinité')
    ax2.set_xlabel('Salinité (PSU)')
    ax2.set_ylabel('Profondeur (m)')
    ax2.legend()
    plt.show()

def plot_sv(sv,depth):
    '''
    This function plot the sound velocity profile from a sv file and a depth files
    Should be the same size
    '''
    fig, ax = plt.subplots(figsize=(6, 5))
    # Tracer la courbe de célérité
    ax.plot(sv, depth, label='Célérité')
    ax.set_xlabel('Célérité (m/s)')
    ax.set_ylabel('Profondeur (m)')
    ax.legend()
    plt.show()

def plot_map_sal_or_temp(i,vect,lat,lon,depth):
    '''
    This function allows to plot on a map the temperature or salinity
    as a function of latitude and longitude at a certain depth.
    if i=1 salinity
    if i=2 Temperature
    '''
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_aspect(aspect='equal')
    im = ax.imshow(vect[depth, :, :], extent=[lon.min(), lon.max(), lat.min(), lat.max()])
    ax.set_xlabel('Longitude (°)')
    ax.set_ylabel('Latitude (°)')
    ax.invert_yaxis()
    if i==1 :
        fig.colorbar(im, ax=ax,label="Salinité (PSU)",fraction=0.94)
    if i==2 :
        fig.colorbar(im, ax=ax,label="Température (°)",fraction=1)
    plt.show()

'''
See source for algorithm choice
Source : Pike, J. M. and F. L. Beiboer, A comparison between algorithms for the speed of sound in seawater,
The Hydrographic Society, Special Publication, 34, 1993.
https://blog.seabird.com/ufaqs/which-algorithm-for-calculating-sound-velocity-sv-from-ctd-data-should-i-use/
less than 1000 meter depth --> Chen-Millero
more than 1000 meter depth --> Delgrosso
'''

def soundspeed(S,T,D):
        # 0 = D
        # # This is copied directly from the UNESCO algorithms.
        # # CHECKVALUE: SVEL=1731.995 M/S, S=40 (IPSS-78),T=40 DEG C,P=10000 DBAR
        # # SCALE PRESSURE TO BARS
        # P = P0 / 10.0
        # SR = np.sqrt(np.abs(S))
        # # S**2 TERM.
        # D = 1.727e-3 - 7.9836e-6 * P
        # # S**3/2 TERM.
        # B1 = 7.3637e-5 + 1.7945e-7 * T
        # B0 = -1.922e-2 - 4.42e-5 * T
        # B = B0 + B1 * P
        # # S**1 TERM.
        # A3 = (-3.389e-13 * T + 6.649e-12) * T + 1.100e-10
        # A2 = ((7.988e-12 * T - 1.6002e-10) * T + 9.1041e-9) * T - 3.9064e-7
        # A1 = (
        #     ((-2.0122e-10 * T + 1.0507e-8) * T - 6.4885e-8) * T - 1.2580e-5
        # ) * T + 9.4742e-5
        # A0 = (((-3.21e-8 * T + 2.006e-6) * T + 7.164e-5) * T - 1.262e-2) * T + 1.389
        # A = ((A3 * P + A2) * P + A1) * P + A0
        # # S**0 TERM.
        # C3 = (-2.3643e-12 * T + 3.8504e-10) * T - 9.7729e-9
        # C2 = (
        #     ((1.0405e-12 * T - 2.5335e-10) * T + 2.5974e-8) * T - 1.7107e-6
        # ) * T + 3.1260e-5
        # C1 = (
        #     ((-6.1185e-10 * T + 1.3621e-7) * T - 8.1788e-6) * T + 6.8982e-4
        # ) * T + 0.153563
        # C0 = (
        #     (((3.1464e-9 * T - 1.47800e-6) * T + 3.3420e-4) * T - 5.80852e-2) * T
        #     + 5.03711
        # ) * T + 1402.388
        # C = ((C3 * P + C2) * P + C1) * P + C0
        # # SOUND SPEED RETURN.
        # ssp = C + (A + B * SR + D * S) * S
        # return ssp

        '''
        Del grosso uses pressure in kg/cm^2.  To get to this from dbars
        we  must divide by "g".  From the UNESCO algorithms (referring to
        ANON (1970) BULLETIN GEODESIQUE) we have this formula for g as a
        function of latitude and pressure.  We set latitude to 45 degrees
        for convenience!
        '''
        XX = np.sin(45 * np.pi / 180)
        GR = 9.780318 * (1.0 + (5.2788e-3 + 2.36e-5 * XX) * XX) + 1.092e-6 * D
        P = D / GR
        # This is from VSOUND.f.
        C000 = 1402.392
        DCT = (0.501109398873e1 - (0.550946843172e-1 - 0.221535969240e-3 * T) * T) * T
        DCS = (0.132952290781e1 + 0.128955756844e-3 * S) * S
        DCP = (0.156059257041e0 + (0.244998688441e-4 - 0.883392332513e-8 * P) * P) * P
        DCSTP = (
            (
                -0.127562783426e-1 * T * S
                + 0.635191613389e-2 * T * P
                + 0.265484716608e-7 * T * T * P * P
                - 0.159349479045e-5 * T * P * P
                + 0.522116437235e-9 * T * P * P * P
                - 0.438031096213e-6 * T * T * T * P
            )
            - 0.161674495909e-8 * S * S * P * P
            + 0.968403156410e-4 * T * T * S
            + 0.485639620015e-5 * T * S * S * P
            - 0.340597039004e-3 * T * S * P
        )
        ssp = C000 + DCT + DCS + DCP + DCSTP
        return ssp

def get_indexlatlon(lat_test,lon_test,lat,lon):
    '''
    Allows to get the indexs of the closest lat and lon to compare
    with a lat and lon array
    '''
    lat_idx = np.abs(lat-lat_cast2).argmin()
    lon_idx = np.abs(lon-lon_cast2).argmin()

    return lat_idx,lon_idx


if __name__ == '__main__' :
    # loading data from GDEM
    data = scipy.io.loadmat('../../data/GDEM/Jun_GDEMV.mat')
    map = scipy.io.loadmat('../../data/GDEM/geomap_GDEMV.mat')

    #getting celerity and salinity and converting no data to nan
    salinity=data['salinity']
    temp=data['water_temp']
    salinity = salinity.astype(float)
    salinity[salinity==-32000]=np.nan
    salinity=salinity*0.001+15
    temp = temp.astype(float)
    temp[temp==-32000]=np.nan
    temp=temp*0.001+15

    #getting lat,lon, depth from map and translate depth to -depth
    lat=map['lat']
    lon=map['lon']
    depth=map['depth']*-1

    #convert lon from 0-360 to -180,180)
    # lon=lon_360to180(lon)


    #import data to compare and get lat, lon of campaign data NMEA
    data_file2 = "../../data/AE2008_CTD/AE2008_PostDeployment_Cast2_deriv.cnv"
    lat_cast2,lon_cast2=extract_lat_lon(data_file2)

    #translate lon to match with GDEM_grid
    lon_cast2=lon_cast2+360

    #get index in GDEM that correspond to lat, lon of the campaign data
    lat_idx,lon_idx=get_indexlatlon(lat_cast2,lon_cast2,lat,lon)

    #get temp, salinity of the right lat and lon
    temp_cast2 = temp[:, lat_idx, lon_idx]
    salinity_cast2 = salinity[:, lat_idx, lon_idx]

    #convert depth into pressure in dB and compute soundspeed with Chen-Millero equation
    pressure=swpressure(depth*-1,lat_cast2)
    sv=soundspeed(salinity_cast2,temp_cast2,pressure[:,0])

    plot_map_sal_or_temp(1,salinity,lat,lon,0)

    plot_sal_temp(salinity_cast2,temp_cast2,depth)
    plot_sv(sv,depth)
