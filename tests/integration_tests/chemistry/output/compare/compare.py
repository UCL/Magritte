import numpy as np
import matplotlib.pyplot as plt
import sys



name = str(sys.argv[1])

if(len(sys.argv) > 2):
    tag  = "_" + str(sys.argv[2])
else:
    tag = ""


file_name = "../files/" + name + tag + ".txt"



my_data  = np.loadtxt(file_name)

ncells = np.shape(my_data)[0]



file_name = "output_3D-PDR/" + name + tag + "_3D-PDR.txt"

their_data = np.loadtxt(file_name)

arow           = their_data[-2]
their_data[-2] = their_data[-1]
their_data[-1] = arow


error          = my_data - their_data
relative_error = 2.0*abs(error)/abs(my_data+their_data)


# Make the plots

print " "
print "Comparing " + name

fig = plt.figure()

ax1 = fig.add_subplot(111)

data_line = np.zeros(ncells)

data_line = relative_error


lambdac = np.array([910.0E0, 950.0E0, 1000.0E0, 1050.0E0, 1110.0E0, 1180.0E0, 1250.0E0, 1390.0E0, 1490.0E0, 1600.0E0, 1700.0E0,  1800.0E0,  1900.0E0,  2000.0E0, 2100.0E0, 2190.0E0,  2300.0E0,  2400.0E0,  2500.0E0, 2740.0E0, 3440.0E0,  4000.0E0,  4400.0E0,  5500.0E0, 7000.0E0, 9000.0E0, 12500.0E0, 22000.0E0, 34000.0E0])
Xc      = np.array([5.76E0, 5.18E0, 4.65E0, 4.16E0, 3.73E0, 3.40E0, 3.11E0, 2.74E0, 2.63E0, 2.62E0, 2.54E0, 2.50E0, 2.58E0, 2.78E0, 3.01E0, 3.12E0, 2.86E0, 2.58E0, 2.35E0, 2.00E0, 1.58E0, 1.42E0, 1.32E0, 1.00E0, 0.75E0, 0.48E0, 0.28E0, 0.12E0, 0.05E0])


# if(max(data_line) > 1.0E-99):
ax1.plot(data_line)
# ax1.plot(data[:75,0],data[:75,1],label="Magritte")
# ax1.plot(data2[:75,0],data2[:75,1],label="3D-PDR")
# ax1.plot(data[:,0],relative_error[:,1])
# ax1.plot(data[:,1])
# ax1.plot(data2[:,1])
# ax1.scatter(lambdac,Xc)


ax1.set_title(name + " relative error")
ax1.set_xlabel("n (grid point nr)")
ax1.set_ylabel(name + "relative error")
ax1.set_yscale("log")
# ax1.set_xscale("log")
ax1.legend()

fig.tight_layout()

plot_name = "error_" + name + ".png"


# Save the plot in pdf format

fig.savefig(plot_name, bbox_inches='tight')

print "Plot saved as " + plot_name
print " "
