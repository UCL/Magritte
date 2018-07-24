import matplotlib.pyplot as plt
import numpy as np

xg,yg,zg, vx,vy,vz, density = np.loadtxt("../../../../input/1D_regular_101.txt", unpack=True)
ncells = np.shape(xg)[0]

data = np.loadtxt('level_populations.txt', skiprows=1)
nlev = np.shape(data)[1]

# mean_intensity_data = np.loadtxt("mean_intensity.txt")
# nrad = np.shape(mean_intensity_data)[0]

# print( np.shape(mean_intensity_data) )

print("ncells =", ncells)
print("nlev =", nlev)

cell = 0


fig = plt.figure()
ax1 = fig.add_subplot(211)

pop = np.zeros(ncells)
mean_intensity = np.zeros(ncells)

print(np.shape(data))

for level in range(nlev):
    for point in range(ncells):
        pop[point] = data[point][level]

    ax1.plot(pop, label=level)

ax1.legend()
ax1.set_xlabel("x (grid point)")
ax1.set_ylabel("level populations")
ax1.set_yscale("log")

ax2 = fig.add_subplot(212)


# for rad in range(nrad):
#     for point in range(ncells):
#
#         mean_intensity[point] = mean_intensity_data[rad][point]
#
#     ax2.plot(mean_intensity, label=rad)

ax2.legend()
ax2.set_xlabel("x (grid point)")
ax2.set_ylabel("mean intensity")
ax2.set_yscale("log")

fig.tight_layout()
fig.savefig("levels.pdf", bbox_inches='tight')

plt.show()
