# Script to plot the temperatures
# -------------------------------



import matplotlib.pyplot as plt
import numpy as np
import sys



# Check the tag of the data that is to be plotted

if (len(sys.argv)>1):
    tag = "_" + str(sys.argv[1])
else:
    tag = ""


# Read the gas temperatures file

file_name = "../temperature_gas" + tag + ".txt"

temperature_gas = np.loadtxt(file_name)
ngrid = np.shape(temperature_gas)[0]


# Read the dust temperatures file

file_name = "../temperature_dust" + tag + ".txt"

temperature_dust = np.loadtxt(file_name)


print "ngrid = " + str(ngrid)


# Make the plots

fig = plt.figure()


pop = np.zeros(ngrid)
mean_intensity = np.zeros(ngrid)


# Plot gas temperatures

ax1 = fig.add_subplot(211)

ax1.plot(temperature_gas)

ax1.legend()
ax1.set_title("temperature gas")
ax1.set_xlabel("x (grid point)")
ax1.set_ylabel("gas temperatures")
# ax1.set_yscale("log")


# Plot dust temperatures

ax2 = fig.add_subplot(212)

ax2.plot(temperature_dust)

ax2.legend()
ax2.set_title("temperature dust")
ax2.set_xlabel("x (grid point)")
ax2.set_ylabel("dust temperatures")
# ax2.set_yscale("log")


fig.tight_layout()

plot_name = "temperatures" + tag + ".pdf"

fig.savefig(plot_name, bbox_inches='tight')


print "Plot saved as " + plot_name


plt.show()
