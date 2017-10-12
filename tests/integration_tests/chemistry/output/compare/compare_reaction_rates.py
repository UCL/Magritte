import numpy as np
import matplotlib.pyplot as plt
import sys



name = str(sys.argv[1])


if(len(sys.argv) > 2):
    tag  = "_" + str(sys.argv[2])
else:
    tag = ""

if(name=="all"):
    name = "reaction"

print "Comparing"

file_name = "../" + name + "_rates" + tag + ".txt"

print file_name

my_rates = np.loadtxt(file_name)

ngrid = np.shape(my_rates)[0]
nreac = np.shape(my_rates)[1]


file_name = "output_3D-PDR/" + name + "_rates" + tag + "_3D-PDR.txt"

print file_name

their_rates = np.loadtxt(file_name)


error          = my_rates - their_rates
relative_error = 2.0*abs(error)/(my_rates+their_rates)


# Make the plots

fig = plt.figure()

ax1 = fig.add_subplot(111)

data_line = np.zeros(ngrid+1)

for reac in range(nreac):
    data_line = relative_error[:,reac]
    if( (np.mean(data_line[2:20]) > 0.9E-6 and np.mean(data_line) < 1.0E10) or False ):
        ax1.plot(data_line, label=int(reac))
        # ax1.plot(my_rates[:,reac], label=int(reac))
        # ax1.plot(their_rates[:,reac], label=int(reac))
    if(False):
        ax1.plot(my_rates[:,reac], label=int(reac))
        ax1.plot(their_rates2[:,reac], label=int(reac))

ax1.legend()
ax1.set_title(name + " relative error")
ax1.set_xlabel("x (grid point)")
ax1.set_ylabel(name + " relative error")
ax1.set_yscale("log")

fig.tight_layout()

plot_name = "error_" + name + tag + ".png"


# Save the plot in pdf format

fig.savefig(plot_name, bbox_inches='tight')

print "Plot saved as " + plot_name
print " "
