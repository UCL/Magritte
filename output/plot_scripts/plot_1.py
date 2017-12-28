# Script to plot the temperatures
# -------------------------------



import matplotlib.pyplot as plt
import numpy as np
import sys



print " "
print "Plot 1D array"
print "-------------"



# Check whether the date stamp of the datafile is given

if (len(sys.argv)>1):
    date_stamp = str(sys.argv[1])
else:
    print "ERROR : No date stamp given !\n"
    print "Please try again and give the date stamp of the output file you want to plot\n"



# Check the tag of the data that is to be plotted

if (len(sys.argv)>2):
    name = str(sys.argv[2])
else:
    print "ERROR : No name given !"
    print "Please try again and give the name of the output file you want to plot\n"

# Check the tag of the data that is to be plotted

if (len(sys.argv)>3):
    tag = "_" + str(sys.argv[3])
else:
    tag = ""



# Read the gas temperatures file

file_name = "../files/" + date_stamp + "_output/" + name + tag + ".txt"

temperature_gas = np.loadtxt(file_name)
ncells = np.shape(temperature_gas)[0]

print "Plotting the data as specified in " + file_name



# Make the plots

fig = plt.figure()



# Plot data

ax1 = fig.add_subplot(111)

ax1.plot(temperature_gas)

ax1.set_title(name)
ax1.set_xlabel("n (grid point nr)")
ax1.set_ylabel(name)




fig.tight_layout()

plot_name = "../files/" + date_stamp + "_output/plots/" + name + tag + ".png"

fig.savefig(plot_name, bbox_inches='tight')


print "Plot saved as " + plot_name
print " "
