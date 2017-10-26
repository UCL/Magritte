# Script to plot the chemical abundances
# --------------------------------------



import matplotlib.pyplot as plt
import numpy as np
import sys



print " "
print "Plot chemical abundances"
print "------------------------"



# Check whether the date stamp of the datafile is given

if (len(sys.argv)>1):
    date_stamp = str(sys.argv[1])
else:
    print "ERROR : No date stamp given !\n"
    print "Please try again and give the date stamp of the output file you want to plot\n"



# Check the tag of the data that is to be plotted

if (len(sys.argv)>2):
    tag = "_" + str(sys.argv[2])
else:
    tag = ""



# Get the input files from parameters.txt

with open("../files/" + date_stamp + "_output/parameters.txt") as parameters_file:
    parameters = parameters_file.readlines()

grid_inputfile = "../../" + parameters[38].split()[0]
spec_datafile  = "../../" + parameters[40].split()[0]



# Read the grid input file

xg,yg,zg, vx,vy,vz, density = np.loadtxt(grid_inputfile, unpack=True)
ngrid                       = np.shape(xg)[0]



# Read the abundances output file

file_name = "../files/" + date_stamp + "_output/abundances" + tag + ".txt"

abundances_data = np.loadtxt(file_name)
nspec           = np.shape(abundances_data)[1]



# Read the species names for the legend

species_name = ["dummy"]

with open(spec_datafile) as spec_file:
    for spec in range(0,nspec-2):
        species_name.append( spec_file.readline().split(",")[1] )


# Helper function
# ---------------
def get_species_nr(name):
    nr = 0
    for spec in species_name:
        if spec == name:
            return nr
        nr=nr+1
    print "WARNING species not found!"
    return 0
# ---------------


# Check if there are specific species to be plotted

if (len(sys.argv)>3):
    species_I_want    = get_species_nr( str(sys.argv[3]) )
    species_specified = True
else:
    species_I_want    = ""
    species_specified = False


# Make the plots

print "Plotting the abundances as specified in " + file_name

fig = plt.figure()

ax1 = fig.add_subplot(111)

abundance = np.zeros(ngrid)


for spec in range(1,nspec-1):
    for point in range(ngrid):
        abundance[point] = abundances_data[point][spec]

    if( ( not species_specified and max(abundance) > 1.0E-10)
          or (species_specified and species_I_want == spec) ):
        ax1.plot(abundance, label=species_name[spec])

ax1.legend()
ax1.set_title("chemical abundances " + tag)
ax1.set_xlabel("n (grid point nr)")
ax1.set_ylabel("abundances")
ax1.grid()
ax1.set_yscale("log")

fig.tight_layout()

plot_name = "../files/" + date_stamp + "_output/plots/abundances" + tag + ".png"



# Save the plot in png format

fig.savefig(plot_name, bbox_inches='tight')

print "Plot saved as " + plot_name
print " "



# Show the plot

fig.show()
