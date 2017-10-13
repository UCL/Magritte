import numpy as np
import matplotlib.pyplot as plt
import sys



name = str(sys.argv[1])

if(len(sys.argv) > 2):
    tag  = "_" + str(sys.argv[2])
else:
    tag = ""


file_name = "../files/" + name + tag + ".txt"

print file_name

data = np.loadtxt(file_name)

ngrid  = np.shape(data)[0]
nindex = np.shape(data)[1]


file_name = "output_3D-PDR/" + name + tag + "_3D-PDR.txt"

data2 = np.loadtxt(file_name)

nrows = np.shape(data2)[0]
ncols = np.shape(data2)[1]

for i in range(nrows):
    temp        = data2[i,0]
    data2[i,0]  = data2[i,4]
    data2[i,4]  = temp
    temp        = data2[i,10]
    data2[i,10] = data2[i,6]
    data2[i,6]  = temp


error          = data - data2
relative_error = 2.0*abs(error)/abs(data+data2)


# Make the plots

print " "
print "Plotting " + file_name


fig1 = plt.figure()
fig2 = plt.figure()

ax1 = fig1.add_subplot(111)
ax2 = fig2.add_subplot(111)

data_line = np.zeros(ngrid)


for index in range(nindex):

    data_line = relative_error[:,index]

    if(np.mean(data_line) > 0.1E-10 or True):

        ax1.plot(data_line, label=index)

        data_line1 = data[:,index]
        ax2.plot(data_line1, label=index)
        data_line2 = data2[:,index]
        ax2.plot(data_line2, label=index)

ax1.legend()
ax1.set_title(name + " error")
ax1.set_xlabel("n (grid point nr)")
ax1.set_ylabel(name + " error")
ax1.grid()
ax1.set_yscale("log")

fig1.tight_layout()


ax2.legend()
ax2.set_title(name + " error")
ax2.set_xlabel("n (grid point nr)")
ax2.set_ylabel(name + " error")
ax2.grid()
ax2.set_yscale("log")

fig2.tight_layout()


plot_name1 = "error_" + name + tag + ".png"
plot_name2 = "both_" + name + tag + ".png"


# Save the plot

fig1.savefig(plot_name1, bbox_inches='tight')
fig2.savefig(plot_name2, bbox_inches='tight')


print "Plots saved as "
print "   " + plot_name1
print "   " + plot_name2
print " "
