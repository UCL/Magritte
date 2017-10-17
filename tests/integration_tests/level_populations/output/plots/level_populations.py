# Script to plot the level populations
# ------------------------------------



import matplotlib.pyplot as plt
import numpy as np
import sys



# Check the tag of the data that is to be plotted

if (len(sys.argv)>1):
    tag = "_" + str(sys.argv[1])
else:
    tag = ""


# Get the input files from parameters.txt

with open("../../parameters.txt") as parameters_file:
    parameters = parameters_file.readlines()

nlspec = int(parameters[45].split()[0])

grid_inputfile = "../../../../../" + parameters[38].split()[0]
line_datafile  = ["../../../../../" + parameters[48+2*lspec].split()[0] for lspec in range(nlspec) ]


# Extract the names of the line producing species fron the datafile names

lspec_name = ["" for lspec in range(nlspec)]

for lspec in range(nlspec):
    lspec_name[lspec] = line_datafile[lspec].split("/")[6]
    lspec_name[lspec] = lspec_name[lspec].split(".")[0]


# Print the results

print "nlspec         : " + str(nlspec)
print "grid inputfile : " + grid_inputfile

for lspec in range(nlspec):
    print "line datafile " + str(lspec) + " for " + lspec_name[lspec] + " : " + line_datafile[lspec]


# Read the grid input file

xg,yg,zg, vx,vy,vz, density = np.loadtxt(grid_inputfile, unpack=True)

ngrid = np.shape(xg)[0]



for lspec in range(nlspec):

    # Read the level populations file

    file_name = "../files/level_populations_" + lspec_name[lspec] + tag + ".txt"

    data = np.loadtxt(file_name)
    nlev = np.shape(data)[1]


    # Read the line intensities file

    file_name = "../files/line_intensities_" + lspec_name[lspec] + tag + ".txt"

    mean_intensity_data = np.loadtxt(file_name)
    nrad = np.shape(mean_intensity_data)[0]


    # Make the plots

    fig = plt.figure()


    pop = np.zeros(ngrid)
    mean_intensity = np.zeros(ngrid)



    # Plot level populations

    ax1 = fig.add_subplot(211)

    for level in range(nlev):
        for point in range(ngrid):
            pop[point] = data[point][level]

        ax1.plot(pop, label=level)

    ax1.legend()
    ax1.set_title("level populations for " + lspec_name[lspec])
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
    ax2.set_title("line intensities for " + lspec_name[lspec])
    ax2.set_xlabel("x (grid point)")
    ax2.set_ylabel("line intensities")
    ax2.set_yscale("log")


    fig.tight_layout()

    plot_name = "level_populations_"+lspec_name[lspec]+".png"

    fig.savefig(plot_name, bbox_inches='tight')


    print "Plot saved as " + plot_name
