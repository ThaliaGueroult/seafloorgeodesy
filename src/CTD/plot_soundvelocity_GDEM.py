import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from swpressure import *
from coordfromHeader import *
from scipy.interpolate import interp1d
from homogeneous_pol_mat import *
from read_GDEM import *


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
    ax1.grid()
    ax1.legend()

    # Tracer la courbe de salinité
    ax2.plot(sal, depth,'g', label='Salinité')
    ax2.set_xlabel('Salinité (PSU)')
    ax2.set_ylabel('Profondeur (m)')
    ax2.legend()
    ax2.grid()
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
    ax.grid()
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
    ax.grid()
    if i==1 :
        fig.colorbar(im, ax=ax,label="Salinité (PSU)",fraction=0.15,shrink=0.5)
    if i==2 :
        fig.colorbar(im, ax=ax,label="Température (°)",fraction=0.15,shrink=0.5)
    plt.show()

'''
See source for model choice sound Velocity
Source : Pike, J. M. and F. L. Beiboer, A comparison between algorithms for the speed of sound in seawater,
The Hydrographic Society, Special Publication, 34, 1993.
https://blog.seabird.com/ufaqs/which-algorithm-for-calculating-sound-velocity-sv-from-ctd-data-should-i-use/
less than 1000 meter depth --> Chen-Millero
more than 1000 meter depth --> Delgrosso
'''

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
        P = D * 1.01972/10
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
#
# def soundspeed_chen(S,T,D):
#     P0 = D
#     # This is copied directly from the UNESCO algorithms.
#     # CHECKVALUE: SVEL=1731.995 M/S, S=40 (IPSS-78),T=40 DEG C,P=10000 DBAR
#     # SCALE PRESSURE TO BARS
#     P = P0 / 10.0
#     SR = np.sqrt(np.abs(S))
#     # S**2 TERM.
#     D = 1.727e-3 - 7.9836e-6 * P
#     # S**3/2 TERM.
#     B1 = 7.3637e-5 + 1.7945e-7 * T
#     B0 = -1.922e-2 - 4.42e-5 * T
#     B = B0 + B1 * P
#     # S**1 TERM.
#     A3 = (-3.389e-13 * T + 6.649e-12) * T + 1.100e-10
#     A2 = ((7.988e-12 * T - 1.6002e-10) * T + 9.1041e-9) * T - 3.9064e-7
#     A1 = (((-2.0122e-10 * T + 1.0507e-8) * T - 6.4885e-8) * T - 1.2580e-5) * T + 9.4742e-5
#     A0 = (((-3.21e-8 * T + 2.006e-6) * T + 7.164e-5) * T - 1.262e-2) * T + 1.389
#     A = ((A3 * P + A2) * P + A1) * P + A0
#     # S**0 TERM.
#     C3 = (-2.3643e-12 * T + 3.8504e-10) * T - 9.7729e-9
#     C2 = (((1.0405e-12 * T - 2.5335e-10) * T + 2.5974e-8) * T - 1.7107e-6) * T + 3.1260e-5
#     C1 = (((-6.1185e-10 * T + 1.3621e-7) * T - 8.1788e-6) * T + 6.8982e-4) * T + 0.153563
#     C0 = ((((3.1464e-9 * T - 1.47800e-6) * T + 3.3420e-4) * T - 5.80852e-2) * T+ 5.03711) * T + 1402.388
#     C = ((C3 * P + C2) * P + C1) * P + C0
#     # SOUND SPEED RETURN.
#     ssp = C + (A + B * SR + D * S) * S
#     return ssp
#
#
# def get_indexlatlon(lat_test,lon_test,lat,lon):
#     '''
#     Allows to get the indexs of the closest lat and lon to compare
#     with a lat and lon array
#     '''
#     lat_idx = np.abs(lat-lat_cast2).argmin()
#     lon_idx = np.abs(lon-lon_cast2).argmin()
#
#     return lat_idx,lon_idx

def sound_velocity_munk(z,z1,B,C1):
    eps = 0.00737
    nu = 2*(z-z1)/B
    return C1*(1.0+eps*(nu-1+np.exp(-nu)))

def diff_plot_soundvelocity(depth,sv,c,sv_delgrosso_cast2,depths_cast2):
    # interpolate the depth and sv arrays to the new depth axis
    sv_interp = interp1d(depth.flatten(), sv, bounds_error=False, fill_value="extrapolate")(depths_cast2[:np.argmin(depths_cast2)])
    c_interp = interp1d(depth.flatten(), c.flatten(), bounds_error=False, fill_value="extrapolate")(depths_cast2[:np.argmin(depths_cast2)])
    # now depth_new, sv_interp, and c_interp have the same length along the interpolation axis

    # Calculate the differences between the profiles
    diff_bm_gdem = sv_delgrosso_cast2[:np.argmin(depths_cast2)] - sv_interp
    diff_bm_munk = sv_delgrosso_cast2[:np.argmin(depths_cast2)] - c_interp
    diff_gdem_munk = sv_interp - c_interp
    print(np.mean(diff_bm_gdem))
    # Plot the differences
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.plot(depths_cast2[:np.argmin(depths_cast2)],diff_bm_gdem,label='Bermuda - GDEM')
    ax.plot(depths_cast2[:np.argmin(depths_cast2)],diff_bm_munk,label="Bermuda - Munk")
    ax.plot(depths_cast2[:np.argmin(depths_cast2)],diff_gdem_munk,label="GDEM - Munk")
    ax.set_ylabel('Sound Velocity Difference (m/s)')
    ax.set_xlabel('Depth (m)')
    ax.legend()
    ax.grid()
    ax.invert_xaxis()
    plt.title('Differences in Sound Velocity Profiles')
    plt.show()

if __name__ == '__main__' :
    print(soundspeed_delgrosso(39,26,145))

    # loading data from GDEM
    map = scipy.io.loadmat('../../data/GDEM/geomap_GDEMV.mat')
    data = scipy.io.loadmat('../../data/GDEM/Jun_GDEMV.mat')

    #import data to compare and get lat, lon of campaign data NMEA
    data_file2 = "../../data/AE2008_CTD/AE2008_PostDeployment_Cast2_deriv.cnv"
    lat_cast2,lon_cast2=extract_lat_lon(data_file2)
    print(lat_cast2,lon_cast2)

    #Uses the lat lon from campaign to extract the point of the GDEM grid
    temp_gdem, sal_gdem, std_temp_gdem, std_sal_gdem, depth_gdem = extract_gdem(lat_cast2, lon_cast2, map, data)
    print(std_temp_gdem)
    #Defining pressure standard deviation at 0
    std_p_gdem=np.zeros_like(std_temp_gdem)

    #convert depth into pressure in dB and compute soundspeed with Del Grosso equation
    pressure=swpressure(depth_gdem,lat_cast2)
    sv_gdem=c_delgrosso(sal_gdem,temp_gdem,pressure)
    std_c_gdem=std_c_delgrosso(sal_gdem, temp_gdem, pressure, std_sal_gdem, std_temp_gdem,std_p_gdem)
    print(std_c_gdem, std_c_gdem.shape)
    std_neg = sv_gdem - std_c_gdem
    std_pos = sv_gdem + std_c_gdem

    sv_munk = sound_velocity_munk(depth_gdem,1508.69,1785.58,1498.32)

    data2 = np.loadtxt(data_file2,skiprows=348)
    sv_cast2 = data2[:,21]
    depths_cast2 = data2[:,0]
    sv_cast2 = sv_cast2[:np.argmax(depths_cast2)+1]
    depths_cast2 = depths_cast2[:np.argmax(depths_cast2)+1]

    # interpolate the depth and sv arrays to the new depth axis
    sv_gdem_interp = interp1d(depth_gdem, sv_gdem, bounds_error=False, fill_value="extrapolate")(depths_cast2)
    sv_munk_interp = interp1d(depth_gdem, sv_munk.flatten(), bounds_error=False, fill_value="extrapolate")(depths_cast2)
    # now depth_new, sv_interp, and c_interp have the same length along the interpolation axis

    # Calculate the differences between the profiles
    diff_bm_gdem = sv_cast2 - sv_gdem_interp
    diff_bm_munk = sv_cast2 - sv_munk_interp
    diff_gdem_munk = sv_gdem_interp - sv_munk_interp

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 15))
    plt.suptitle('Sound Velocity Profile Bermuda Comparison')

    ax1.plot(sv_cast2, depths_cast2,'green', label='Bermuda, Princeton 2020')
    ax1.set_xlabel('Sound Velocity (m/s)')
    ax1.set_ylabel('Depth (m)')
    # ax1.legend()
    ax1.grid()
    ax1.invert_yaxis()
    ax1.set_title('Bermuda, 2020, Princeton')

    ax2.plot(sv_gdem, depth_gdem,'navy', label="GDEM, Navy Prediction")
    ax2.set_xlabel('Sound Velocity (m/s)')
    # ax2.legend()
    ax2.grid()
    ax2.invert_yaxis()
    ax2.set_title('GDEM, US Navy, Prediction')

    ax3.plot(sv_cast2, depths_cast2,'green', label='Bermuda, Princeton 2020')
    ax3.plot(sv_gdem, depth_gdem,'navy', label="GDEM, Navy 2003")
    ax3.fill_betweenx(depth_gdem, std_neg, std_pos, facecolor='orange', alpha=0.5, label='Standard Deviation of GDEM')
    ax3.set_xlabel('Sound Velocity (m/s)')
    ax3.legend()
    ax3.grid()
    ax3.invert_yaxis()
    ax3.set_title('Comparison of profiles')

    ax4.plot(diff_bm_gdem, depths_cast2, label='Bermuda - GDEM')
    ax4.fill_betweenx(depth_gdem, - std_c_gdem, std_c_gdem, facecolor='orange', alpha=0.5, label='Standard Deviation of GDEM')
    ax4.set_xlabel('Sound Velocity Difference (m/s)')
    ax4.set_xlim(-np.nanmax(std_c_gdem)*1.1,np.nanmax(std_c_gdem)*1.1)
    ax4.legend()
    ax4.grid()
    ax4.invert_yaxis()
    ax4.set_title('Differences in Sound Velocity Profiles')
    plt.show()

    # fig = plt.figure(figsize=(16, 10))
    # gs = fig.add_gridspec(2,2)
    # ax1 = fig.add_subplot(gs[0, 0])
    # ax2 = fig.add_subplot(gs[0, 1])
    # ax3 = fig.add_subplot(gs[1, :])
    #
    # ax1.plot(sv_delgrosso_cast2, depths_cast2, label='Bermuda, Princeton 2020')
    # ax1.set_xlabel('Sound Velocity (m/s)')
    # ax1.set_ylabel('Depth (m)')
    # ax1.legend()
    # ax1.grid()
    # ax1.set_title('Sound Velocity Profile Bermuda, 2020')
    #
    # # Premier tracé à gauche
    # ax2.plot(sv_delgrosso_cast2, depths_cast2, label='Bermuda, Princeton 2020')
    # ax2.plot(sv, depth, label="GDEM, Navy 2003")
    # ax2.plot(c, depth, label='Munk Model, 1974')
    # ax2.set_xlabel('Sound Velocity (m/s)')
    # ax2.set_ylabel('Depth (m)')
    # ax2.legend()
    # ax2.grid()
    # ax2.set_title('Comparison of Sound Velocity Profile')
    #
    # # Deuxième tracé à droite
    # ax3.plot(depths_cast2[:np.argmin(depths_cast2)], diff_bm_gdem,'red', label='Bermuda - GDEM')
    # ax3.plot(depths_cast2[:np.argmin(depths_cast2)], diff_bm_munk, 'purple',label="Bermuda - Munk")
    # ax3.plot(depths_cast2[:np.argmin(depths_cast2)], diff_gdem_munk, 'cyan', label="GDEM - Munk")
    # ax3.set_ylabel('Sound Velocity Difference (m/s)')
    # ax3.set_xlabel('Depth (m)')
    # ax3.legend()
    # ax3.grid()
    # ax3.invert_xaxis()
    # ax3.set_title('Differences in Sound Velocity Profiles')
    #
    # # Ajuster l'espace entre les deux tracés
    # plt.subplots_adjust(wspace=0.3)
    #
    # # Afficher la figure
    # plt.show()

    # fig = plt.figure(figsize=(16, 10))
    # gs = fig.add_gridspec(2,2)
    # ax1 = fig.add_subplot(gs[0, 0])
    # ax2 = fig.add_subplot(gs[0, 1])
    # ax3 = fig.add_subplot(gs[1, :])
    #
    # ax1.plot(sv_cast2, depths_cast2,'green', label='Bermuda, Princeton 2020')
    # ax1.set_xlabel('Sound Velocity (m/s)')
    # ax1.set_ylabel('Depth (m)')
    # ax1.legend()
    # ax1.grid()
    # ax1.set_title('Sound Velocity Profile Bermuda, 2020, Princeton')
    #
    # ax2.plot(sv_gdem, depth_gdem,'navy', label="GDEM, Navy 2003")
    # ax2.set_xlabel('Sound Velocity (m/s)')
    # ax2.set_ylabel('Depth (m)')
    # ax2.legend()
    # ax2.grid()
    # ax2.set_title('Sound Velocity Profile Bermuda, 2003, GDEM US Navy')
    # ax3.scatter(diff_bm_gdem, depths_cast2, label='Bermuda - GDEM')
    # ax3.set_ylabel('Sound Velocity Difference (m/s)')
    # ax3.set_xlabel('Depth (m)')
    # ax3.legend()
    # ax3.grid()
    # ax3.invert_xaxis()
    # ax3.set_title('Differences in Sound Velocity Profiles')
    #
    # # Ajuster l'espace entre les deux tracés
    # plt.subplots_adjust(wspace=0.3)
    #
    # # Afficher la figure
    # plt.show()

    # plot_map_sal_or_temp(1,salinity,lat,lon,0)
    # plot_sal_temp(salinity_cast2,temp_cast2,depth)
    # plot_sv(sv,depth)

    # fig, ax = plt.subplots(figsize=(16, 10))
    # ax.plot(sv_delgrosso_cast2,depths_cast2,label='Bermuda, Princeton 2020')
    # ax.plot(sv,depth,label="GDEM, Navy 2003")
    # ax.plot(c, depth,label='Munk Model, 1979')
    # ax.set_xlabel('Sound Velocity (m/s)')
    # ax.set_ylabel('Depth (m)')
    # ax.legend()
    # ax.grid()
    # plt.title('Comparison of Sound Velocity Profile')
    # plt.show()
    #
    # temp_delgrosso_cast2 = data2[:,1]
    # fig, ax = plt.subplots(figsize=(16, 10))
    # ax.plot(temp_delgrosso_cast2[:np.argmin(depths_cast2)],depths_cast2[:np.argmin(depths_cast2)],label='Bermuda, Princeton 2020')
    # ax.plot(temp[:, lat_idx, lon_idx],depth,label="GDEM, Navy 2003")
    # ax.set_xlabel('Temperature (°)')
    # ax.set_ylabel('Depth (m)')
    # ax.legend()
    # ax.grid()
    # plt.title('Comparison of temperature Bermuda 2003 - 2020')
    # plt.show()
    #
    # sal_delgrosso_cast2 = data2[:,4]
    # fig, ax = plt.subplots(figsize=(16, 10))
    # ax.plot(sal_delgrosso_cast2[:np.argmin(depths_cast2)],depths_cast2[:np.argmin(depths_cast2)],label='Bermuda, Princeton 2020')
    # ax.plot(salinity[:, lat_idx, lon_idx],depth,label="GDEM, Navy 2003")
    # ax.set_xlabel('salinity (PSU)')
    # ax.set_ylabel('Depth (m)')
    # ax.legend()
    # ax.grid()
    # plt.title('Comparison of salinity Bermuda 2003 - 2020')
    # plt.show()

    # diff_plot_soundvelocity(depth,sv,c,sv_delgrosso_cast2,depths_cast2)
