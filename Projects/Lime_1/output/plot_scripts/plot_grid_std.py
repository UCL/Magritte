import sys
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D


# ---------------
# Plotgrid points
# ---------------


# Check whether the date stamp of the datafile is given
if (len(sys.argv) > 1):
    date_stamp = str(sys.argv[1])
else:
    print "ERROR : No date stamp given !\n"
    print "Please try again and give the date stamp of the output file you want to plot\n"


# Check the tag of the data that is to be plotted
if (len(sys.argv)>2):
    tag = "_" + str(sys.argv[2])
else:
    tag = ""


# Read data
file_name1 = "../files/{}_output/grid{}.txt".format(date_stamp, tag)
file_name2 = "../files/{}_output/grid_full{}.txt".format(date_stamp, tag)


ID, x,y,z, vx,vy,vz, density = np.loadtxt(file_name1, unpack = True)

IDf, xf,yf,zf, vxf,vyf,vzf, densityf = np.loadtxt(file_name2, unpack = True)


NCELLS  = len(ID)
NCELLSf = len(IDf)

print "number of cells {}".format(NCELLS)


# PLOT
fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')


# Plot grid points
ax1.scatter(xf, yf, zf, alpha=0.3)
ax1.scatter(x, y, z, color="red", alpha=1)


# Add cell numbers to plot
for i in range(NCELLSf):
    ax1.text(xf[i], yf[i], zf[i], i)


# for cell in range(1):
# 	for r in range(NRAYS):
# 		if (radius[cell][r] > 0.0):
# 			lx = xg[cell] + uhpx[r] * np.linspace(0,radius[cell][r],2)
# 			ly = yg[cell] + uhpy[r] * np.linspace(0,radius[cell][r],2)
# 			lz = zg[cell] + uhpz[r] * np.linspace(0,radius[cell][r],2)
#
# 			# Plot the rays
# 			ax1.plot(lx, ly, lz, linewidth=.5, color=".13")


ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_zlabel("z")


fig.tight_layout()
# fig.savefig("grid.pdf", bbox_inches='tight')


plt.show()
