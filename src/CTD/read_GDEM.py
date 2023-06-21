import numpy as np
import scipy.io

# loading data from GDEM
map = scipy.io.loadmat('../../data/GDEM/geomap_GDEMV.mat')
data = scipy.io.loadmat('../../data/GDEM/Jun_GDEMV.mat')
lat = 31.44
lon = -68.693

def extract_gdem(lat, lon, map, data):
    #in GDEM_grid, lat from [-89,90] and lon from [0, 359.75]
    lon = lon % 360.0
    print(lat,lon)

    #getting lat,lon, depth from map and translate depth to -depth
    lat_GDEM = map['lat']
    lon_GDEM = map['lon']
    depth_GDEM = map['depth'][:,0]

    # get index in GDEM that correspond to the latitude and longitude searched
    lat_idx = np.abs(lat_GDEM-lat).argmin()
    lon_idx = np.abs(lon_GDEM-lon).argmin()

    #extract salinity, temperatude, salinity standard deviation and temperature standard deviation from GDEM grid
    salinity, temp = data['salinity'], data['water_temp']
    std_sal, std_temp=data['salinity_stdev'], data['water_temp_stdev']

    # for a value I in the GDEM_grid, to get adjusted value V :
    # V = add_offset + scale_factor * I
    # with scale_factor = 0.001 and add_offset = 15
    # if no value (32000), replaces value by np.nan
    salinity = salinity.astype(float)
    salinity[salinity == - 32000] = np.nan
    salinity = salinity * 0.001 + 15

    temp = temp.astype(float)
    temp[temp == - 32000] = np.nan
    temp=temp * 0.001 + 15

    std_sal = std_sal.astype(float)
    std_sal[std_sal == -32000] = np.nan
    std_sal = std_sal * 0.001 + 15

    std_temp = std_temp.astype(float)
    std_temp[std_temp == -32000] = np.nan
    std_temp = std_temp * 0.001 + 15

    #get temp, salinity, std_temp and std_sal of the right lat and lon
    temp_gdem = temp[:, lat_idx, lon_idx]
    sal_gdem = salinity[:, lat_idx, lon_idx]
    std_temp_gdem = std_temp[:, lat_idx, lon_idx]
    std_sal_gdem = std_sal[:, lat_idx, lon_idx]

    return (temp_gdem, sal_gdem, std_temp_gdem, std_sal_gdem, depth_GDEM)
