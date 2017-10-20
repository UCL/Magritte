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

ngrid = np.shape(my_data)[0]



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

data_line = np.zeros(ngrid)

data_line = relative_error




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
