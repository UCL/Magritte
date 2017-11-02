# Script to plot the temperatures
# -------------------------------



import matplotlib.pyplot as plt
import numpy as np
import sys



print " "
print "Plot 2 1D arrays"
print "----------------"



# Check whether the date stamp of the datafile is given

if (len(sys.argv)>1):
    date_stamp = str(sys.argv[1])
else:
    print "ERROR : No date stamp given !\n"
    print "Please try again and give the date stamp of the output file you want to plot\n"



# Check the name of the data that is to be plotted

if (len(sys.argv)>2):
    name = str(sys.argv[2])
else:
    print "ERROR : No name given !"
    print "Please try again and give the name of the output file you want to plot\n"


# Check the second name of the data that is to be plotted

if (len(sys.argv)>3):
    name2 = str(sys.argv[3])
else:
    print "ERROR : No second name given !"
    print "Please try again and give the second name of the output file you want to plot\n"



# Read the data file

file_name = "../files/" + date_stamp + "_output/" + name + ".txt"

data = np.loadtxt(file_name)
ngrid = np.shape(data)[0]

print "Plotting the data as specified in " + file_name



# Read the second data file

file_name2 = "../files/" + date_stamp + "_output/" + name2 + ".txt"

data2 = np.loadtxt(file_name2)
ngrid = np.shape(data2)[0]

print "Plotting the data as specified in " + file_name2


# Make the plots

fig = plt.figure()



# Plot data

ax1 = fig.add_subplot(111)

ax1.plot(data)
ax1.plot(data2)

ax1.set_title(name)
ax1.set_xlabel("n (grid point nr)")
ax1.set_ylabel(name)




fig.tight_layout()

plot_name = "../files/" + date_stamp + "_output/plots/" + name + ".png"

fig.savefig(plot_name, bbox_inches='tight')


print "Plot saved as " + plot_name
print " "
