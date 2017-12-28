import numpy as np
import matplotlib.pyplot as plt

from pylab import *


from mpl_toolkits.mplot3d import Axes3D



# -----------------------------
# Plot results from Ray Tracing
# -----------------------------



xg,yg,zg = np.loadtxt("input/incells.txt", unpack=True)
ncells = np.shape(xg)[0]

uhpx, uhpy, uhpz = np.loadtxt("output/healpix.txt", unpack=True)
NRAYS = np.shape(uhpx)[0]

#Z, ray = np.loadtxt("Output/eval.txt", unpack=True)

#key = np.loadtxt("Output/key.txt")

#cum_raytot = np.loadtxt("Output/cum_raytot.txt")

mean_intensity = np.loadtxt("output/mean_intensity.txt")


def numray(point, ray, rnr):
	return key[point][rnr + cum_raytot[point][ray]]



xe=np.zeros(ncells*ncells)
ye=np.zeros(ncells*ncells)
ze=np.zeros(ncells*ncells)

radius = np.zeros((ncells,NRAYS))


# PLOT

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


colmap = cm.ScalarMappable(cmap='afmhot')
colmap.set_array(mean_intensity)

pl = ax.scatter(xg, yg, zg, c=cm.afmhot(mean_intensity), marker='o')
cb = fig.colorbar(colmap)



ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")


fig.tight_layout()
fig.savefig("field.pdf", bbox_inches='tight')

plt.show()

