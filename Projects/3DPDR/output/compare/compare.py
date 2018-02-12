import numpy as np
import matplotlib.pyplot as plt
import sys


# Check whether the date stamp of the datafile is given
if (len(sys.argv)>1):
    date_stamp = str(sys.argv[1])
else:
    print("ERROR : No date stamp given !\n")
    print("Please try again and give the date stamp of the output file you want to plot\n")


name = str(sys.argv[2])


if(len(sys.argv) > 3):
    tag  = "_" + str(sys.argv[3])
else:
    tag = ""


file_name = "../files/" + date_stamp + "_output/" + name + tag + ".txt"


my_data = np.loadtxt(file_name)
ncells  = np.shape(my_data)[0]


file_name = "output_3D-PDR/1Dn30/" + name + tag + "_3D-PDR.txt"

their_data = np.loadtxt(file_name)

arow           = their_data[-2]
their_data[-2] = their_data[-1]
their_data[-1] = arow

nrows = np.shape(their_data)[0]



error          = my_data - their_data
relative_error = 2.0*abs(error)/abs(my_data+their_data)


# Make the plots
print(" ")
print("Plotting " + file_name)


fig1 = plt.figure()
fig2 = plt.figure()

ax1 = fig1.add_subplot(111)
ax2 = fig2.add_subplot(111)

data_line = np.zeros(ncells)
data_line = relative_error[:]
ax1.plot(data_line)

data_line1 = my_data[:]
ax2.plot(data_line1)
data_line2 = their_data[:]
ax2.plot(data_line2)

# ax1.legend()
ax1.set_title(name + tag + " error")
ax1.set_xlabel("n (grid point nr)")
ax1.set_ylabel(name + " error")
ax1.grid()
# ax1.set_yscale("log")

fig1.tight_layout()


ax2.legend()
ax2.set_title("both " + name + tag)
ax2.set_xlabel("n (grid point nr)")
ax2.set_ylabel(name + " error")
ax2.grid()
# ax2.set_yscale("log")

fig2.tight_layout()


plot_name1 = "../files/" + date_stamp + "_output/plots/error_" + name + tag + ".png"
plot_name2 = "../files/" + date_stamp + "_output/plots/both_" + name + tag + ".png"


# Save the plot
fig1.savefig(plot_name1, bbox_inches='tight')
fig2.savefig(plot_name2, bbox_inches='tight')


print("Plots saved as  ")
print("   " + plot_name1)
print("   " + plot_name2)
print("                ")
