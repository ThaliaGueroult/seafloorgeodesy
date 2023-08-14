import numpy as np


# global variables : parameters for ellipsoid WGS84
a = 6378137
f = 1/298.257223563
e2 = 2*f - f**2

def deg2rad(ang):
    return ang*np.pi/180.0

def rad2deg(ang):
    return ang*180.0/np.pi

def geod2ecef(lon, lat, hgt):
    x, y, z = np.nan, np.nan, np.nan
    lon, lat, hgt = deg2rad(lon), deg2rad(lat), hgt
    v = np.sqrt(1-e2*np.sin(lat)**2)
    N = a/v
    x = (N+hgt)*np.cos(lon)*np.cos(lat)
    y = (N+hgt)*np.sin(lon)*np.cos(lat)
    z = (N*(1-e2)+hgt)*np.sin(lat)
    return x, y, z

def ecef2geod(x, y, z):
    f = 1-np.sqrt(1-e2)
    r = np.sqrt(x**2+y**2+z**2)
    mu = np.arctan(z/np.sqrt(x**2+y**2)*((1-f)+a*e2/r))
    lon = np.arctan(y/x)
    lat = np.arctan((z*(1-f)+e2*a*np.sin(mu)**3)/((1-f)*(np.sqrt(x**2+y**2)-e2*a*np.cos(mu)**3)))
    hgt = np.sqrt(x**2+y**2)*np.cos(lat)+z*np.sin(lat)-a*np.sqrt(1-e2*np.sin(lat)**2)
    return rad2deg(lon),rad2deg(lat),hgt

def enu2geod(lon0, lat0, hgt0, e, n, u):
    lon, lat, hgt = np.nan, np.nan, np.nan
    x0, y0, z0 = geod2ecef(lon0, lat0, hgt0)
    lon0 = deg2rad(lon0)
    lat0 = deg2rad(lat0)
    R = np.array([[-np.sin(lon0), np.cos(lon0), 0],
                  [-np.sin(lat0) * np.cos(lon0), -np.sin(lat0) * np.sin(lon0), np.cos(lat0)],
                  [np.cos(lat0) * np.cos(lon0), np.cos(lat0) * np.sin(lon0), np.sin(lat0)]])
    enu = np.array([e, n, u]).T
    v0 = np.array([x0, y0, z0]).T
    v = (np.linalg.inv(R) @ enu + v0)
    lon, lat, hgt = ecef2geod(v[0], v[1], v[2])
    return lon, lat, hgt

def geod2enu(lon0, lat0, hgt0, lon, lat, hgt):
    e, n, u = np.nan, np.nan, np.nan
    x, y, z = geod2ecef(lon, lat, hgt)
    x0, y0, z0 = geod2ecef(lon0,lat0,hgt0)
    lon0 = deg2rad(lon0)
    lat0 = deg2rad(lat0)
    R = np.array([[-np.sin(lon0), np.cos(lon0), 0],
                  [-np.sin(lat0)*np.cos(lon0), -np.sin(lat0)*np.sin(lon0), np.cos(lat0)],
                  [np.cos(lat0)*np.cos(lon0), np.cos(lat0)*np.sin(lon0), np.sin(lat0)]])
    v = np.array([x, y, z]).T
    v0 = np.array([x0, y0, z0]).T
    enu = (R@(v-v0))
    return enu[0],enu[1],enu[2]


def utm_mu(n, lon, lat):
    mu = np.nan
    vprim = np.sqrt(1 + ep2 ** 2 * np.cos(lat) ** 2)
    lamda0 = deg2rad(l * (n - 31) + 3)
    mu = utm_k0 * (1 + vprim ** 2 / 2 * (lon - lamba0) ** 2 * np.cos(lat) ** 2)
    return mu


def utm_gamma(n, lon, lat):
    gamma = np.nan
    lamda0 = deg2rad(l * (n - 31) + 3)
    gamma = (lon - lambda0) * np.sin(lat) * (1 + ((lon - lambda0) ** 2 * np.cos(lat) ** 2) / 3)
    return gamma


def utm_geod2map(n, lon, lat):
    if lat > 0:
        Y0 = 0
    else:
        Y0 = 10000000
    k0 = 0.9996
    lambda0 = 6 * (n - 31) + 3
    b = beta(lat)
    lon, lat, lon0 = deg2rad(lon), deg2rad(lat), deg2rad(lon0)
    ep2 = e2 / (1 - e2)
    vp2 = 1 + ep2 * np.cos(lat) ** 2
    n1 = np.sqrt(1 + ep2 * np.cos(lat) ** 4)
    v = np.sqrt(1 - e2 * np.sin(lat) ** 2)
    N = a / v
    rho = a * (1 - e2) / v ** 3
    X = X0 + k0 * np.sqrt(rho * N) / 2 * np.log((n1 + np.sqrt(vp2) * np.cos(lat) * np.sin(n1 * (lon - lon0))) / (
            n1 - np.sqrt(vp2) * np.cos(lat) * np.sin(n1 * (lon - lon0))))
    Y = Y0 + k0 * b + k0 * np.sqrt(rho * N) * (
            np.arctan(np.tan(lat) / (np.sqrt(vp2) * np.cos(n1 * (lon - lon0)))) - np.arctan(
        np.tan(lat) / np.sqrt(vp2)))
    return X, Y


def utm_map2geod(n, X, Y):
    lon, lat = np.nan, np.nan
    # ...
    return lon, lat


def beta(lat):
    beta = np.nan
    lat = deg2rad(lat)
    e = np.sqrt(e2)
    b0 = -175 / 16384 * e ** 8 - 5 / 256 * e ** 6 - 3 / 64 * e ** 4 - 1 / 4 * e ** 2 + 1
    b1 = -105 / 4096 * e ** 8 - 45 / 1024 * e ** 6 - 3 / 32 * e ** 4 - 3 / 8 * e ** 2
    b2 = 525 / 16384 * e ** 8 + 45 / 1024 * e ** 6 + 15 / 256 * e ** 4
    b3 = -175 / 12288 * e ** 8 - 35 / 3072 * e ** 6
    b4 = 315 / 131072 * e ** 8
    beta = a * (b0 * lat + b1 * np.sin(2 * lat) + b2 * np.sin(4 * lat) + b3 * np.sin(6 * lat) + b4 * np.sin(8 * lat))
    return beta


def geod2iso(lat):
    lat_iso = np.nan
    # ...
    return lat_iso


def iso2geod(lat_iso):
    lat = np.nan
    # ...
    return lat

if __name__ == "__main__":
    print("Geodesy Tollbox")
