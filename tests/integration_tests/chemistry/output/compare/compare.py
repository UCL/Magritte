import numpy as np
import matplotlib.pyplot as plt
import sys



name = str(sys.argv[1])

if(len(sys.argv) > 2):
    tag  = "_" + str(sys.argv[2])
else:
    tag = ""


file_name = "../" + name + tag + ".txt"

print file_name

data  = np.loadtxt(file_name)

ngrid = np.shape(data)[0]



file_name = "output_3D-PDR/" + name + tag + "_3D-PDR.txt"

data2 = np.loadtxt(file_name)


error          = data - data2
relative_error = 2.0*abs(error)/(data+data2)


# Make the plots

print " "
print "Plotting " + file_name

fig = plt.figure()

ax1 = fig.add_subplot(111)

data_line = np.zeros(ngrid)

data_line = relative_error
# if(max(data_line) > 1.0E-99):
ax1.plot(data_line)


ax1.set_title(name + " relative error")
ax1.set_xlabel("x (grid point)")
ax1.set_ylabel(name + "relative error")
ax1.set_yscale("log")

fig.tight_layout()

plot_name = "error_" + name + ".png"


# Save the plot in pdf format

fig.savefig(plot_name, bbox_inches='tight')

print "Plot saved as " + plot_name
print " "
