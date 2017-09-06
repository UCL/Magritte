# Script to plot the level populations
# ------------------------------------



import matplotlib.pyplot as plt
import numpy as np



# Get the input files from parameters.txt

with open("../../parameters.txt") as parameters_file:
    parameters = parameters_file.readlines()

nlspec = int(parameters[45].split()[0])

grid_inputfile = "../" + parameters[38].split()[0]
line_datafile  = "../" + parameters[48].split()[0]

print "nlspec         : " + str(nlspec)
print "grid inputfile : " + grid_inputfile
print "line datafile  : " + line_datafile


# Read the grid input file

xg,yg,zg, vx,vy,vz, density = np.loadtxt(grid_inputfile, unpack=True)

ngrid = np.shape(xg)[0]





# Read the level populations file

data = np.loadtxt('level_populations.txt', skiprows=1)
nlev = np.shape(data)[1]


# Read the mean intensity file

mean_intensity_data = np.loadtxt("mean_intensities.txt")
nrad = np.shape(mean_intensity_data)[0]


# Make the plots

fig = plt.figure()


pop = np.zeros(ngrid)
mean_intensity = np.zeros(ngrid)

lspec_name = line_datafile.split("/")[2]
lspec_name = lspec_name.split(".")[0]


# Plot level populations

ax1 = fig.add_subplot(211)

for level in range(nlev):
    for point in range(ngrid):
        pop[point] = data[point][level]

    ax1.plot(pop, label=level)

ax1.legend()
ax1.set_title("level populations for " + lspec_name)
ax1.set_xlabel("x (grid point)")
ax1.set_ylabel("level populations")
ax1.set_yscale("log")


# Plot mean intensity

ax2 = fig.add_subplot(212)

for rad in range(nrad):
    for point in range(ngrid):

        mean_intensity[point] = mean_intensity_data[rad][point]

    ax2.plot(mean_intensity, label=rad)

ax2.legend()
ax2.set_title("mean intensities for " + lspec_name)
ax2.set_xlabel("x (grid point)")
ax2.set_ylabel("mean intensities")
ax2.set_yscale("log")


fig.tight_layout()

plot_name = "level_populations_"+lspec_name+".pdf"

fig.savefig(plot_name, bbox_inches='tight')


print "Plot saved as " + plot_name


plt.show()
