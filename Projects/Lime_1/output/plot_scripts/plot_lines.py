# Script to plot level populations
# --------------------------------


import matplotlib.pyplot as plt
import numpy as np
import sys


print "                                           "
print "Plot level populations and line intensities"
print "-------------------------------------------"


# Check whether date stamp of datafile is given
if (len(sys.argv)>1):
    date_stamp = str(sys.argv[1])
else:
    print "ERROR : No date stamp given !\n"
    print "Please try again and give the date stamp of the output file you want to plot\n"


# Check species of lines that are to be plotted
if (len(sys.argv)>2):
    lspec = str(sys.argv[2])
else:
    print "ERROR : No line producing species given !\n"
    print "Please try again and give the name of the species for which you want to plot the lines\n"


# Check tag of data that is to be plotted
if (len(sys.argv)>3):
    tag = "_" + str(sys.argv[3])
else:
    tag = ""


# Get input files from parameters.hpp
with open("../../parameters.hpp") as parameters_file:
    for line in parameters_file:
        line = line.split()
        if len(line) is 3:
            if line[1] == 'INPUTFILE':
                inputfile = "../../" + line[2].split("\"")[1]


# Read grid input file
ID, xg,yg,zg, vx,vy,vz, density = np.loadtxt(inputfile, unpack=True)
ncells = np.shape(xg)[0]



# For all line data files


# Read level populations file
file_name = "../files/" + date_stamp + "_output/level_populations_" + lspec + tag + ".txt"

data = np.loadtxt(file_name)
nlev = np.shape(data)[1]



# Make plots

fig1 = plt.figure()

pop = np.zeros(ncells)



# Plot level populations
ax1 = fig1.add_subplot(111)

for level in range(nlev):
    for point in range(ncells):
        pop[point] = data[point,level]

    ax1.plot(pop, label=level)

ax1.legend()
ax1.set_title("level populations for " + lspec)
ax1.set_xlabel("n (grid point nr)")
ax1.set_ylabel("level populations")
ax1.set_yscale("log")

fig1.tight_layout()

plot_name = "../files/" + date_stamp + "_output/plots/level_populations_"+lspec+ tag +".png"

fig1.savefig(plot_name, bbox_inches='tight')


# For all line data files


# Read line intensities file
file_name = "../files/" + date_stamp + "_output/line_intensities_" + lspec + tag + ".txt"

mean_intensity_data = np.loadtxt(file_name)
nrad = np.shape(mean_intensity_data)[1]



# Make plots
fig2 = plt.figure()

mean_intensity = np.zeros(ncells)



# Plot line intensity
ax2 = fig2.add_subplot(111)

for rad in range(nrad):
    for point in range(ncells):

        mean_intensity[point] = mean_intensity_data[point][rad]

    ax2.plot(mean_intensity, label=rad)

ax2.legend()
ax2.set_title("line intensities for " + lspec)
ax2.set_xlabel("n (grid point nr)")
ax2.set_ylabel("line intensities")
ax2.set_yscale("log")


fig2.tight_layout()

plot_name = "../files/" + date_stamp + "_output/plots/line_intensities_" + lspec + tag + ".png"

fig2.savefig(plot_name, bbox_inches='tight')


print "Plot " + str(lspec) + " saved as " + plot_name

print " "
