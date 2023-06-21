import numpy as np

def swpressure(dorp=None, lat=30, dtop=1):
    """
    From Frederik J.Simons, Princeton, transcripted in Python
    Pressure/depth in seawater of a global ocean following
    Saunders (1981) "Practical conversion of pressure to depth"
    10.1175/1520-0485(1981)011<0573:PCOPTD>2.0.CO;2

    INPUT:

    dorp     Depth(s), in positive meters down from the surface, OR:
            Pressure(s), in decibar (=1e4 Pa)
    lat      Latitude(s), in decimal degrees
    dtop     1 Input is depth and output is pressure [default]
            2 Input is pressure and output is depth

    OUTPUT:

    pord     Pressure, in decibar (=1e4 Pa), OR:
            Depth(s), in positive meters down from the surface, OR:

    EXAMPLE:

    x = np.random.rand()*1000
    lat = np.random.rand()*180-90
    np.testing.assert_almost_equal(swpressure(swpressure(x, lat, 1), lat, 2), x)

    SEE ALSO:

    RDGDEM3.f by Michael Carnes, Naval Oceanographic Office (2002)
    SW_PRES.f by Phil Morgan, CSIRO (1993)
    CALPRESSURE.m by Yifeng Wang, Princeton (2010)
    """

    # Default - the result should be 7500 db give or take
    if dorp is None:
        dorp = 7321.45

    # Note that both inputs must be equal sized or one of them scalar
    dorp = np.atleast_1d(dorp)
    lat = np.atleast_1d(lat)

    # The Saunders parameters c1 [m/db] and c2 [m/db^2]
    c1 = (5.92 + 5.25*np.sin(np.abs(lat)*np.pi/180)**2)*1e-3
    c2 = 2.21*1e-6

    # The equation in m and decibar is the quadratic in 2
    if dtop == 1:
        # Depth to pressure via the quadratic equation solution
        pord = ((1-c1) - np.sqrt((1-c1)**2 - 4*c2*dorp))/(2*c2)
    elif dtop == 2:
        # Pressure to depth
        pord = (1-c1)*dorp - c2*dorp**2

    # Variable output
    return pord
