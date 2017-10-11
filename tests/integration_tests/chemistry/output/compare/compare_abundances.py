import numpy as np
import matplotlib.pyplot as plt
import sys



if(len(sys.argv) > 1):
    tag  = "_" + str(sys.argv[1])
else:
    tag = ""


file_name = "../abundances" + tag + ".txt"

print file_name

my_abn = np.transpose(np.loadtxt(file_name))

ngrid = np.shape(my_abn)[0]
nspec = np.shape(my_abn)[1]


file_name = "output_3D-PDR/abundances" + tag + "_3D-PDR.txt"

their_abn = np.loadtxt(file_name)


error          = my_abn - their_abn
relative_error = 2.0*abs(error)/abs(my_abn+their_abn)


# Make the plots

print " "
print "Plotting " + file_name

fig = plt.figure()

ax1 = fig.add_subplot(111)

data_line = np.zeros(ngrid+1)

for spec in range(nspec):
    data_line = relative_error[:,spec]
    if( (np.mean(data_line) > 0.7E-10 and np.mean(data_line) < 1.0E10) or False ):
        ax1.plot(data_line, label=spec)
        # ax1.plot(my_abn[:,spec], label=spec)
        # ax1.plot(their_abn[:,spec], label=spec)
    if(False):
        ax1.plot(my_abn[:,spec], label=spec)
        ax1.plot(their_abn2[:,spec], label=spec)

ax1.legend()
ax1.set_title("abundances relative error")
ax1.set_xlabel("x (grid point)")
ax1.set_ylabel("abundances relative error")
ax1.set_yscale("log")

fig.tight_layout()

plot_name = "error_abundances.png"


# Save the plot in pdf format

fig.savefig(plot_name, bbox_inches='tight')

print "Plot saved as " + plot_name
print " "
