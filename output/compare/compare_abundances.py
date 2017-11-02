import numpy as np
import matplotlib.pyplot as plt
import sys



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


# Get the input files from parameters.hpp

with open("../../parameters.hpp") as parameters_file:
    for line in parameters_file:
        line = line.split()
        if len(line) is 3:
            if line[1] == 'SPEC_DATAFILE':
                spec_datafile = "../../" + line[2].split("\"")[1]


my_file_name = "../files/" + date_stamp + "_output/abundances" + tag + ".txt"


my_abn = np.loadtxt(my_file_name)

ngrid = np.shape(my_abn)[0]
nspec = np.shape(my_abn)[1]



file_name = "output_3D-PDR/1Dn30/abundances" + tag + "_3D-PDR.txt"

their_abn = np.loadtxt(file_name)

abn             = their_abn[-2,:]
their_abn[-2,:] = their_abn[-1,:]
their_abn[-1,:] = their_abn[-2,:]


error          = my_abn - their_abn
relative_error = 2.0*abs(error)/abs(my_abn+their_abn)


# Read the species names for the legend

species_name = ["dummy"]

with open(spec_datafile) as spec_file:
    for spec in range(1,nspec-1):
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



# Check if there are specific species to be plotted

if (len(sys.argv)>3):
    species_I_want = get_species_nr( str(sys.argv[2]) )
    species_specified = True
else:
    species_I_want = ""
    species_specified = False



# Make the plots

print " "
print "Plotting for" + file_name
print "and " + my_file_name

fig1 = plt.figure()
fig2 = plt.figure()

ax1 = fig1.add_subplot(111)
ax2 = fig2.add_subplot(111)

data_line = np.zeros(ngrid+1)

for spec in range(nspec):
    data_line = relative_error[:,spec]
    if( (np.mean(data_line) > 0.7E-30 and not species_specified) or (species_I_want == spec and species_specified) ):
        ax1.plot(data_line, label=species_name[spec])
        ax2.plot(my_abn[:,spec], label=species_name[spec])
        ax2.plot(their_abn[:,spec], label=species_name[spec])


ax1.legend()
ax1.set_title("abundances " + tag +" relative error")
ax1.set_xlabel("x (grid point)")
ax1.set_ylabel("abundances relative error")
ax1.grid()
ax1.set_yscale("log")

ax2.legend()
ax2.set_title("Both abundances " + tag)
ax2.set_xlabel("x (grid point)")
ax2.set_ylabel("abundances relative error")
ax2.grid()
ax2.set_yscale("log")

fig1.tight_layout()
fig2.tight_layout()

plot_name1 = "../files/" + date_stamp + "_output/plots/error_abundances" + tag + ".png"
plot_name2 = "../files/" + date_stamp + "_output/plots/both_abundances" + tag + ".png"



# Save the plot in pdf format

fig1.savefig(plot_name1, bbox_inches='tight')
fig2.savefig(plot_name2, bbox_inches='tight')


print "Plots saved as "
print "   " + plot_name1
print "   " + plot_name2
print " "
