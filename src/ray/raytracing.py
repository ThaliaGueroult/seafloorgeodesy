import numpy as np
import scipy
import matplotlib.pyplot as plt

#Code from Vincent, 2001

def rayz(c, z, zA, zB, iga):
    """
    Perform ray tracing from depth A (source) to depth B (receiver)
    with given initial grazing angle and sound velocity profile.

    The speed of sound is assumed to vary only with depth and the
    sound rays suffer neither reversals nor reflections.

    Args:
        c (array): sound velocity profile (column vector)
        z (array): depth array for the sound velocity profile
        zA (float): depth of point A
        zB (float): depth of point B
        iga (float): initial grazing angle (in degrees)

    Returns:
        hd (float): horizontal distance
        tt (float): transit time
        ve (float): effective sound velocity (slant range = ve * tt)
    """
    # Effective range for sound velocity profile
    cs, zs = effvp(c, z, zA, zB)

    # Determine critical incident (grazing) angle
    grzcri = crtgz(cs, zs, zA)

    if iga < grzcri:
        print(f"critical grazing angle = {grzcri} degrees")
        print(f"incident grazing angle = {iga} degrees")
        print("grazing angle is too small so that ray reversal will occur")
        hd = np.nan
        ve = np.nan
        tt = np.nan
    else:
        # No reversal
        cA = np.interp(zA, z, c)
        theA = 90 - iga  # degree between the ray and vertical axis (at z= zA)
        a = np.sin(theA * np.pi / 180) / cA  # Snell's constant (Clay and Medwin's)

        # Ordinary refraction analysis
        if a > 0:
            dz = np.diff(zs)
            b = np.diff(cs) / dz  # gradients

            theta = np.arcsin(a * cs)
            zgi = np.where(np.abs(b) <= 1e-10)[0]  # zero gradient indices
            nzgi = np.where(b != 0)[0]  # non-zero gradient indices

            # Non-zero gradient elements
            R = 1 / (a * b[nzgi])
            ct1 = np.cos(theta[nzgi])
            ct2 = np.cos(theta[nzgi+1])
            dx = np.zeros_like(zs)
            dt = np.zeros_like(zs)
            w1 = cs[nzgi] / b[nzgi]
            w2 = dz[nzgi] + w1
            dx[nzgi] = (ct1 - ct2) * R
            dt[nzgi] = 1 / b[nzgi] * np.log((w2 * (1 + ct1)) / (w1 * (1 + ct2)))

            # Zero gradient elements
            dx[zgi] = dz[zgi] * np.tan(theta[zgi])
            dt[zgi] = np.sqrt(dx[zgi]**2 + dz[zgi]**2) / cs[zgi]

            tt = np.sum(dt)
            hd = np.sum(dx)

            # If ray plot is desired
            # x0 = 0
            # x = np.cumsum([x0, dx])
            # plt.plot(x, -zs, '.')
            # plt.grid()

            ve = np.sqrt(hd**2 + (zA - zB)**2) / tt

        # Rays propagate vertically
        elif a == 0:
            hd = 0
            ve = harvel(cs, zs)  # harmonic mean sound velocity
            tt = np.abs(zB - zA) / ve

        else:
            raise ValueError('Use incident angle between 0 and 90 degrees')

    return hd, tt, ve

if __name__ == '__main__' :
    data = np.loadtxt('../../dat/cast2.dat')
    cz=data[:,1]
    z=data[:np.argmax(cz),0]
    cz=data[:np.argmax(cz),1]

    rayz(cz, z, 0, np.argmax(cz), 45)
