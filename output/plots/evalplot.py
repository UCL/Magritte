import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D



# -----------------------------
# Plot results from Ray Tracing
# -----------------------------



xg,yg,zg, vx,vy,vz = np.loadtxt("../input/grid.txt", unpack=True)
ngrid = np.shape(xg)[0]

uhpx, uhpy, uhpz = np.loadtxt("healpix.txt", unpack=True)
NRAYS = np.shape(uhpx)[0]

Z, ray, tr = np.loadtxt("eval.txt", unpack=True)

key = np.loadtxt("key.txt")

cum_raytot = np.loadtxt("cum_raytot.txt")

raytot = np.zeros((ngrid,NRAYS))

for n1 in range(ngrid):
	raytot[n1][0] = cum_raytot[n1][1]
	for r1 in range(1,NRAYS-1):
		raytot[n1][r1] = cum_raytot[n1][r1+1] - cum_raytot[n1][r1]

print(raytot)


def numray(point, ray, rnr):
	return key[point][rnr + cum_raytot[point][ray]]



xe=np.zeros(ngrid*ngrid)
ye=np.zeros(ngrid*ngrid)
ze=np.zeros(ngrid*ngrid)

radius = np.zeros((ngrid,NRAYS))


# PLOT

fig = plt.figure()
ax = fig.add_subplot(121, projection='3d')

# Plot the gridpoints
ax.scatter(xg, yg, zg)

for gridpoint in range(1):

	for gp in range(ngrid):

		p = gridpoint*ngrid + gp

		if (Z[p] > 0):

			xe[p] = xg[gridpoint] + uhpx[ray[p]] * Z[p]
			ye[p] = yg[gridpoint] + uhpy[ray[p]] * Z[p]
			ze[p] = zg[gridpoint] + uhpz[ray[p]] * Z[p]

		if (1.1*Z[p] > radius[gridpoint][ray[p]]):
			radius[gridpoint][ray[p]] = 1.1*Z[p]
	

# Plot the evaluation points
ax.scatter(xe.tolist(), ye.tolist(), ze.tolist())
		
for gridpoint in range(1):
	for r in range(NRAYS):
		if (radius[gridpoint][r] > 0.0):
			lx = xg[gridpoint] + uhpx[r] * np.linspace(0,radius[gridpoint][r],2)
			ly = yg[gridpoint] + uhpy[r] * np.linspace(0,radius[gridpoint][r],2)
			lz = zg[gridpoint] + uhpz[r] * np.linspace(0,radius[gridpoint][r],2)

			# Plot the rays
			ax.plot(lx, ly, lz, linewidth=.5, color=".13")
		

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")


fig.tight_layout()
fig.savefig("plots.pdf", bbox_inches='tight')



ax2 = fig.add_subplot(122, projection='3d')

xe2=np.zeros(ngrid*ngrid)
ye2=np.zeros(ngrid*ngrid)
ze2=np.zeros(ngrid*ngrid)

radius2 = np.zeros((ngrid,NRAYS))


# Plot the gridpoints
# ax2.scatter(xg, yg, zg)

# for gridpoint in range(1):

# 	for raynr in range(7,8):

# 		for rnr in range(raytot.astype(int)[gridpoint][raynr]):

# 			print("yup")

# 			gp = numray(gridpoint,raynr,rnr)

# 			print(gp)

# 			p = gridpoint*ngrid + gp

# 			if (Z[p] > 0):

# 				xe2[p] = xg[gridpoint] + uhpx[ray[p]] * Z[p]
# 				ye2[p] = yg[gridpoint] + uhpy[ray[p]] * Z[p]
# 				ze2[p] = zg[gridpoint] + uhpz[ray[p]] * Z[p]

# 			if (1.1*Z[p] > radius2[gridpoint][ray[p]]):
# 				radius2[gridpoint][ray[p]] = 1.1*Z[p]
	

# # Plot the evaluation points
# ax2.scatter(xe2.tolist(), ye2.tolist(), ze2.tolist())
		
# for gridpoint in range(ngrid):
# 	for r in range(NRAYS):
# 		if (radius2[gridpoint][r] > 0.0):
# 			lx = xg[gridpoint] + uhpx[r] * np.linspace(0,radius2[gridpoint][r],2)
# 			ly = yg[gridpoint] + uhpy[r] * np.linspace(0,radius2[gridpoint][r],2)
# 			lz = zg[gridpoint] + uhpz[r] * np.linspace(0,radius2[gridpoint][r],2)

# 			# Plot the rays
# 			ax2.plot(lx, ly, lz, linewidth=.5, color=".13")
		

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")


fig.tight_layout()
fig.savefig("plots.pdf", bbox_inches='tight')

plt.show()

