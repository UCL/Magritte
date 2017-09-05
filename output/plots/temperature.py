# Script to plot the temperatures
# -------------------------------



import matplotlib.pyplot as plt
import numpy as np



# Read the gas temperatures file

temperature_gas = np.loadtxt("temperature_gas.txt")
ngrid = np.shape(temperature_gas)[0]


# Read the dust temperatures file

temperature_dust = np.loadtxt("temperature_dust.txt")


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

plot_name = "temperature.pdf"

fig.savefig(plot_name, bbox_inches='tight')


print "Plot saved as " + plot_name


plt.show()
