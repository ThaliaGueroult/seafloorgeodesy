import numpy as np
import matplotlib.pyplot as plt

def gpsdiagram(dist12=4.8, dist13=12.2, dist14=10.1, dist23=10.3, dist24=11.3, dist34=7.1):
    # Determine GPS relative coordinates relative to receiver 4
    gps = np.array([[0, dist14],
                    [dist12, dist14],
                    [dist34, 0],
                    [0, 0]])

    # Line-Line Intersection
    D = (gps[0,0]-gps[2,0])*(gps[1,1]-gps[3,1]) - (gps[0,1]-gps[2,1])*(gps[1,0]-gps[3,0])
    xo = ((gps[0,0]*gps[2,1]-gps[0,1]*gps[2,0])*(gps[1,0]-gps[3,0]) - (gps[0,0]-gps[2,0])*(gps[1,0]*gps[3,1]-gps[1,1]*gps[3,0]))/D
    yo = ((gps[0,0]*gps[2,1]-gps[0,1]*gps[2,0])*(gps[1,1]-gps[3,1]) - (gps[0,1]-gps[2,1])*(gps[1,0]*gps[3,1]-gps[1,1]*gps[3,0]))/D

    pct = 0.05

    # Check the distances
    if abs(dist24**2 - (dist12**2 + dist14**2)) <= pct*dist24**2 and abs(dist13**2 - (dist14**2 + dist34**2)) <= pct*dist13**2:
        plt.figure()
        plt.plot(gps[[0, 1, 2, 3, 0],0], gps[[0, 1, 2, 3, 0],1], 'k')
        plt.plot(gps[[0, 2, 1, 3],0], gps[[0, 2, 1, 3],1], 'k')

        # Plot the GPS stations
        for index in range(4):
            plt.scatter(gps[index,0], gps[index,1], marker='o', edgecolors='k', facecolors='grey', s=100)
            plt.text(gps[index,0], gps[index,1], str(index+1), fontsize=10, ha='center', va='center')

        # Nautical terms (for illustration)
        terms = ['bow', 'stern', 'port', 'starboard']
        offsets = [[0, 3], [0, -5], [-1.5, 0], [1.25, 0]]
        aligns = ['center', 'center', 'right', 'left']

        for index, term in enumerate(terms):
            plt.text(xo + offsets[index][0], yo + offsets[index][1], term, ha=aligns[index], fontsize=12)

        plt.xlabel('distance [m]')
        plt.ylabel('distance [m]')
        plt.grid(True)
        plt.show()
    else:
        print('Cannot compute configuration currently')

# Example Usage
gpsdiagram()
