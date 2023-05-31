import numpy as np
import matplotlib.pyplot as plt
import scipy

def ray(x0, cz, depth, theta0):
    z=np.linspace(0,np.max(depth),100)
    f=scipy.interpolate.interp1d(depth,cz,'cubic')
    sv=f(z)
    depth=z
    cz=sv
    gradc = np.diff(cz) / np.diff(depth)
    gradc = np.insert(gradc, 0, gradc[0])
    v = [cz[0]]
    x = [x0]
    z = [depth[0]]
    t = [0]
    xk=0
    x=[xk]
    zk=0
    z=[0]
    tk=0
    t=[tk]
    theta0=(90-theta0)*np.pi/180
    p = np.sin(theta0) / v[0]
    for k in range(1,len(cz)):
        v.append(v[k-1] + gradc[k] * (depth[k]-depth[k-1]))
        xk += 1 / (p * gradc[k]) * (np.sqrt(1 - p**2 * v[k-1]**2) - np.sqrt(1 - p**2 * v[k]**2))
        x.append(xk)
        zk += 1 / (p * gradc[k]) * (np.sqrt(1 - (p * gradc[k] * x[k] - np.cos(theta0))**2) - p * v[k-1])
        z.append(zk)
        tk+=1 / gradc[k] * np.log((v[k] / v[k-1]) * ((1 + np.sqrt(1 - p**2 * v[k-1]**2)) / (1 + np.sqrt(1 - p**2 * v[k]**2))))
        t.append(tk)
    plt.plot(x,z)
    plt.gca().invert_yaxis()
    plt.show()
    return x, z, t



if __name__=='__main__':
    sv=np.loadtxt('../../data/sv_GDEM.txt')
    depth=-np.loadtxt('../../data/depth_GDEM.txt')
    x,z,t=ray(0,sv,depth,10)
