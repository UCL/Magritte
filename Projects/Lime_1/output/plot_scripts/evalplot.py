import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D



# -----------------------------
# Plot results from Ray Tracing
# -----------------------------


# Get the input files from parameters.txt

with open("../../parameters.txt") as parameters_file:
    parameters = parameters_file.readlines()

inputfile = "../../../../../" + parameters[38].split()[0]


xg,yg,zg, vx,vy,vz, density = np.loadtxt(inputfile, unpack=True)
ncells = np.shape(xg)[0]

uhpx, uhpy, uhpz = np.loadtxt("../healpix.txt", unpack=True)
NRAYS = np.shape(uhpx)[0]

Z, ray, tr = np.loadtxt("../eval.txt", unpack=True)

key = np.loadtxt("../key.txt")

cum_raytot = np.loadtxt("../cum_raytot.txt")

raytot = np.zeros((ncells,NRAYS))

for n1 in range(ncells):
	raytot[n1][0] = cum_raytot[n1][1]
	for r1 in range(1,NRAYS-1):
		raytot[n1][r1] = cum_raytot[n1][r1+1] - cum_raytot[n1][r1]

print(raytot)


def numray(point, ray, rnr):
	return key[point][rnr + cum_raytot[point][ray]]



xe=np.zeros(ncells*ncells)
ye=np.zeros(ncells*ncells)
ze=np.zeros(ncells*ncells)

radius = np.zeros((ncells,NRAYS))


# PLOT

fig = plt.figure()
ax = fig.add_subplot(121, projection='3d')

# Plot the cells
ax.scatter(xg, yg, zg)

for cell in range(1):

	for gp in range(ncells):

		p = cell*ncells + gp

		if (Z[p] > 0):

			xe[p] = xg[cell] + uhpx[ray[p]] * Z[p]
			ye[p] = yg[cell] + uhpy[ray[p]] * Z[p]
			ze[p] = zg[cell] + uhpz[ray[p]] * Z[p]

		if (1.1*Z[p] > radius[cell][ray[p]]):
			radius[cell][ray[p]] = 1.1*Z[p]


# Plot the evaluation points
ax.scatter(xe.tolist(), ye.tolist(), ze.tolist())

for cell in range(1):
	for r in range(NRAYS):
		if (radius[cell][r] > 0.0):
			lx = xg[cell] + uhpx[r] * np.linspace(0,radius[cell][r],2)
			ly = yg[cell] + uhpy[r] * np.linspace(0,radius[cell][r],2)
			lz = zg[cell] + uhpz[r] * np.linspace(0,radius[cell][r],2)

			# Plot the rays
			ax.plot(lx, ly, lz, linewidth=.5, color=".13")


ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")


fig.tight_layout()
fig.savefig("plots.pdf", bbox_inches='tight')



ax2 = fig.add_subplot(122, projection='3d')

xe2=np.zeros(ncells*ncells)
ye2=np.zeros(ncells*ncells)
ze2=np.zeros(ncells*ncells)

radius2 = np.zeros((ncells,NRAYS))


# Plot the cells
# ax2.scatter(xg, yg, zg)

# for cell in range(1):

# 	for raynr in range(7,8):

# 		for rnr in range(raytot.astype(int)[cell][raynr]):

# 			print("yup")

# 			gp = numray(cell,raynr,rnr)

# 			print(gp)

# 			p = cell*ncells + gp

# 			if (Z[p] > 0):

# 				xe2[p] = xg[cell] + uhpx[ray[p]] * Z[p]
# 				ye2[p] = yg[cell] + uhpy[ray[p]] * Z[p]
# 				ze2[p] = zg[cell] + uhpz[ray[p]] * Z[p]

# 			if (1.1*Z[p] > radius2[cell][ray[p]]):
# 				radius2[cell][ray[p]] = 1.1*Z[p]


# # Plot the evaluation points
# ax2.scatter(xe2.tolist(), ye2.tolist(), ze2.tolist())

# for cell in range(ncells):
# 	for r in range(NRAYS):
# 		if (radius2[cell][r] > 0.0):
# 			lx = xg[cell] + uhpx[r] * np.linspace(0,radius2[cell][r],2)
# 			ly = yg[cell] + uhpy[r] * np.linspace(0,radius2[cell][r],2)
# 			lz = zg[cell] + uhpz[r] * np.linspace(0,radius2[cell][r],2)

# 			# Plot the rays
# 			ax2.plot(lx, ly, lz, linewidth=.5, color=".13")


ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")


fig.tight_layout()
fig.savefig("evalplot.png", bbox_inches='tight')

plt.show()
