import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D



uhpx, uhpy, uhpz = np.loadtxt("Output/healpix.txt", unpack=True)
NRAYS = np.shape(uhpx)[0]



antipod = np.zeros(NRAYS)

succes = 0

for n in range(NRAYS):

	for m in range(NRAYS):

			if (uhpx[n]==-uhpx[m] and uhpy[n]==-uhpy[m] and uhpz[n]==-uhpz[m]):

				succes = succes + 1
				antipod[n] = m
				print(n, m)

print(" ")
print("Number of antipodal points ", succes)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

r=6

lx =  uhpx[r] * np.linspace(0,1,2)
ly =  uhpy[r] * np.linspace(0,1,2)
lz =  uhpz[r] * np.linspace(0,1,2)

# Plot the ray
ax.plot(lx, ly, lz, linewidth=.5, color=".13")

lx =  uhpx[int(antipod[r])] * np.linspace(0,1,2)
ly =  uhpy[int(antipod[r])] * np.linspace(0,1,2)
lz =  uhpz[int(antipod[r])] * np.linspace(0,1,2)

# Plot the ray
ax.plot(lx, ly, lz, linewidth=.5, color=".13")

plt.show()
