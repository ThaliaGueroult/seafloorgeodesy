import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from swpressure import *
from coordfromHeader import *
from scipy.interpolate import interp1d


def lon_360to180(lon):
    '''
    This function allows to convert longitude from 0 to 360°
    to -180 to 180°
    '''
    lon=((lon - 180) % 360) - 180
    return(lon)

def plot_sal_temp(sal,temp,depth):
    '''
    This function plot the map of salinity or temperature at a certain depth
    depth is given is GDEM data and is define on 78 levels :

    '''
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
    ax.plot(sv, depth, label='Delgrosso Model')
    ax.set_xlabel('Sound Velocity (m/s)')
    ax.set_ylabel('Depth (m)')
    ax.set_title('Sound Velocity profile from GDEM data')
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
        fig.colorbar(im, ax=ax,label="Salinité (PSU)",fraction=0.15,shrink=0.5)
    if i==2 :
        fig.colorbar(im, ax=ax,label="Température (°)",fraction=0.15,shrink=0.5)
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

def get_indexlatlon(lat_test,lon_test,lat,lon):
    '''
    Allows to get the indexs of the closest lat and lon to compare
    with a lat and lon array
    '''
    lat_idx = np.abs(lat-lat_cast2).argmin()
    lon_idx = np.abs(lon-lon_cast2).argmin()

    return lat_idx,lon_idx

def sound_velocity(z):
    eps = 0.00737
    nu = 2*(z-1300)/1300
    return 1492*(1.0+eps*(nu-1+np.exp(-nu)))

def diff_plot_soundvelocity(depth,sv,c,sv_delgrosso_cast2,depths_cast2):
    # interpolate the depth and sv arrays to the new depth axis
    sv_interp = interp1d(depth.flatten(), sv, bounds_error=False, fill_value="extrapolate")(depths_cast2[:np.argmin(depths_cast2)])
    c_interp = interp1d(depth.flatten(), c.flatten(), bounds_error=False, fill_value="extrapolate")(depths_cast2[:np.argmin(depths_cast2)])
    # now depth_new, sv_interp, and c_interp have the same length along the interpolation axis

    # Calculate the differences between the profiles
    diff_bm_gdem = sv_delgrosso_cast2[:np.argmin(depths_cast2)] - sv_interp
    diff_bm_munk = sv_delgrosso_cast2[:np.argmin(depths_cast2)] - c_interp
    diff_gdem_munk = sv_interp - c_interp
    # Plot the differences
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.plot(depths_cast2[:np.argmin(depths_cast2)],diff_bm_gdem,label='Bermuda - GDEM')
    ax.plot(depths_cast2[:np.argmin(depths_cast2)],diff_bm_munk,label="Bermuda - Munk")
    ax.plot(depths_cast2[:np.argmin(depths_cast2)],diff_gdem_munk,label="GDEM - Munk")
    ax.set_ylabel('Sound Velocity Difference (m/s)')
    ax.set_xlabel('Depth (m)')
    ax.legend()
    ax.invert_xaxis()
    plt.title('Differences in Sound Velocity Profiles')
    plt.show()



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

    # plot_sal_temp(salinity_cast2,temp_cast2,depth)
    # plot_sv(sv,depth)

    c = sound_velocity(depth*-1)
    data2 = np.loadtxt(data_file2,skiprows=348)
    sv_delgrosso_cast2 = data2[:,21]
    depths_cast2 = -data2[:,0]
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.plot(sv_delgrosso_cast2,depths_cast2,label='Bermuda, Princeton 2020')
    ax.plot(sv,depth,label="GDEM, Navy 2003")
    ax.plot(c, depth,label='Munk Model, 1979')
    ax.set_xlabel('Sound Velocity (m/s)')
    ax.set_ylabel('Depth (m)')
    ax.legend()
    plt.title('Comparison of Sound Velocity Profile')
    plt.show()

    temp_delgrosso_cast2 = data2[:,1]
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.plot(temp_delgrosso_cast2[:np.argmin(depths_cast2)],depths_cast2[:np.argmin(depths_cast2)],label='Bermuda, Princeton 2020')
    ax.plot(temp[:, lat_idx, lon_idx],depth,label="GDEM, Navy 2003")
    ax.set_xlabel('Temperature (°)')
    ax.set_ylabel('Depth (m)')
    ax.legend()
    plt.title('Comparison of temperature Bermuda 2003 - 2020')
    plt.show()

    sal_delgrosso_cast2 = data2[:,4]
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.plot(sal_delgrosso_cast2[:np.argmin(depths_cast2)],depths_cast2[:np.argmin(depths_cast2)],label='Bermuda, Princeton 2020')
    ax.plot(salinity[:, lat_idx, lon_idx],depth,label="GDEM, Navy 2003")
    ax.set_xlabel('salinity (PSU)')
    ax.set_ylabel('Depth (m)')
    ax.legend()
    plt.title('Comparison of salinity Bermuda 2003 - 2020')
    plt.show()

    diff_plot_soundvelocity(depth,sv,c,sv_delgrosso_cast2,depths_cast2)
