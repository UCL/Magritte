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

ax1 = fig1.add_subplot(111)

data_line = np.zeros(ngrid)

for index in range(nindex):
    if(index==0 or True):
        # data_line = relative_error[:,index]
        # if(np.mean(data_line) > 1.0E-1 or True):
        #     ax1.plot(data_line, label=index)
        data_line1 = data[:,index]
        ax1.plot(data_line1, label=index)
        data_line2 = data2[:,index]
        ax1.plot(data_line2, label=index)

ax1.legend()
ax1.set_title(name + " error")
ax1.set_xlabel("x (grid point)")
ax1.set_ylabel(name + " error")
ax1.set_yscale("log")

fig1.tight_layout()

plot_name = "error_" + name + tag + ".png"


# Save the plot in pdf format

fig1.savefig(plot_name, bbox_inches='tight')

print "Plot saved as " + plot_name
print " "


# fig2 = plt.figure(2)
#
# ax1 = fig2.add_subplot(121)
#
# data_line = np.zeros(ngrid)
#
# for index in range(nindex):
#     data_line = relative_error[:,index]
#     ax1.plot(data_line, label=index)
#
# ax1.legend()
# ax1.set_title(name + " error")
# ax1.set_xlabel("x (grid point)")
# ax1.set_ylabel(name + " error")
# ax1.set_yscale("log")
#
# fig2.tight_layout()
#
# plot_name = "error_" + name + ".png"
#
#
# # Save the plot in pdf format
#
# fig2.savefig(plot_name, bbox_inches='tight')
