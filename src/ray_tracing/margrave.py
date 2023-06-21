import numpy as np
import matplotlib.pyplot as plt
import scipy

def ray(x0, cz, depth, theta0):
    z=np.linspace(0,5200,1000)
    f=scipy.interpolate.interp1d(depth,cz,'cubic')
    sv=f(z)
    depth=z
    cz=sv
    gradc = np.diff(cz) / np.diff(depth)
    gradc = np.insert(gradc, 0, gradc[0])
    v = [cz[0]]
    xk, tk = x0, depth[0], 0
    x, z, t = [xk], [zk], [tk]
    theta0=(90-theta0)*np.pi/180
    p = np.sin(theta0) / v[0]
    for k in range(1,len(cz)):
        v.append(v[k-1] + gradc[k] * (depth[k]-depth[k-1]))
        xk += 1 / (p * gradc[k]) * (np.sqrt(1 - p**2 * v[k-1]**2) - np.sqrt(1 - p**2 * v[k]**2))
        x.append(xk)
        tk+=1 / gradc[k] * np.log((v[k] / v[k-1]) * ((1 + np.sqrt(1 - p**2 * v[k-1]**2)) / (1 + np.sqrt(1 - p**2 * v[k]**2))))
        t.append(tk)
    plt.plot(x,depth,label='{}°'.format(theta0*180/np.pi))
    plt.title("Ray Path in Linear Wavespeed profile")
    plt.xlabel("Horizontal Distance (km)")
    plt.ylabel("Depth (km)")
    plt.legend()
    plt.gca().invert_yaxis()
    plt.show()
    return x, t

if __name__=='__main__':
    sv=np.loadtxt('../../data/sv_GDEM.txt')
    depth=-np.loadtxt('../../data/depth_GDEM.txt')
    x,z,t=ray(0,sv,depth,10)
