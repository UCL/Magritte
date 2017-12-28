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

inputfile = "../../../../../" + parameters[38].split()[0]
spec_datafile  = "../../../../../" + parameters[40].split()[0]


print "Plot chemical abundances"

print "grid inputfile : " + inputfile


# Read the grid input file

xg,yg,zg, vx,vy,vz, density = np.loadtxt(inputfile, unpack=True)
ncells = np.shape(xg)[0]


# Read the abundances file

file_name = "../abundances" + tag + ".txt"

abundances_data = np.loadtxt("../abundances.txt")
nspec = np.shape(abundances_data)[0]

print "ncells is " + str(ncells)
print "nspec is " + str(nspec)


# Read the species

species_name = ["dummy"]

with open(spec_datafile) as spec_file:
    for spec in range(1,nspec):
        species_name.append( spec_file.readline().split(",")[1] )


# ----------------------------------------------------------------------------------------------- #

# Helper function

def get_species_nr(name):
    nr = 0
    for spec in species_name:
        if spec == name:
            return nr
        nr=nr+1
    print "WARNING species not found!"
    return 0

# ----------------------------------------------------------------------------------------------- #

print "specnr " + str(get_species_nr("H3+"))
print "specnr " + str(get_species_nr("e-"))



# Make the plots

fig = plt.figure()


# Plot abundances

ax1 = fig.add_subplot(111)

abundance = np.zeros(ncells)

for spec in range(nspec):
    for point in range(ncells):
        abundance[point] = abundances_data[spec][point]

    if( max(abundance) > 1.0E-5 ):
        ax1.plot(abundance, label=species_name[spec])

ax1.legend()
ax1.set_title("chemical abundances")
ax1.set_xlabel("x (grid point)")
ax1.set_ylabel("abundances")
ax1.set_yscale("log")



fig.tight_layout()

plot_name = "abundances.pdf"

fig.savefig(plot_name, bbox_inches='tight')


print "Plot saved as " + plot_name


plt.show()
