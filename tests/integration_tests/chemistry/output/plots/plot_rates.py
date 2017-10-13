import numpy as np
import matplotlib.pyplot as plt
import sys



name = str(sys.argv[1])

if (len(sys.argv)>2):
    tag = "_" + str(sys.argv[2])
else:
    tag = ""


file_name = "../files/" + name + tag + ".txt"

data = np.loadtxt(file_name)

ngrid  = np.shape(data)[0]
nindex = np.shape(data)[1]



# Make the plots

print " "
print "Plotting " + file_name

fig = plt.figure()

ax1 = fig.add_subplot(111)

data_line = np.zeros(ngrid)

for index in range(nindex):
    data_line = data[:,index]
    if max(data_line > 1.0E-10):
        ax1.plot(data_line, label=index)

ax1.legend()
ax1.set_title(name)
ax1.set_xlabel("n (grid point nr)")
ax1.set_ylabel(name)
ax1.grid()
ax1.set_yscale("log")

fig.tight_layout()

plot_name = name + tag + ".png"



# Save the plot in pdf format

fig.savefig(plot_name, bbox_inches='tight')

print "Plot saved as " + plot_name
print " "
