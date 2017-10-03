import numpy as np
import matplotlib.pyplot as plt
import sys



name = str(sys.argv[1])

if(len(sys.argv) > 2):
    tag  = "_" + str(sys.argv[2])
else:
    tag = ""


file_name = "../" + name + "_rates" + tag + ".txt"

print file_name

rates = np.loadtxt(file_name)

ngrid = np.shape(rates)[0]
nreac = np.shape(rates)[1]


file_name = name + "_rates" + tag + "_3D-PDR.txt"

rates_2 = np.loadtxt(file_name)


error = rates - rates_2



# Make the plots

print " "
print "Plotting " + file_name

fig = plt.figure()

ax1 = fig.add_subplot(111)

data_line = np.zeros(ngrid)

for reac in range(nreac):
    data_line = data[:,reac]
    if max(data_line > 1.0E-20):
        ax1.plot(data_line, label=reac)

ax1.legend()
ax1.set_title("Error")
ax1.set_xlabel("x (grid point)")
ax1.set_ylabel(name)
ax1.set_yscale("log")

fig.tight_layout()

plot_name = name + ".png"



# Save the plot in pdf format

fig.savefig(plot_name, bbox_inches='tight')

print "Plot saved as " + plot_name
print " "
