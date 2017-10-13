import numpy as np
import matplotlib.pyplot as plt
import sys



if(len(sys.argv) > 1):
    tag  = "_" + str(sys.argv[1])
else:
    tag = ""


file_name = "../files/abundances" + tag + ".txt"

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

fig1 = plt.figure()
fig2 = plt.figure()

ax1 = fig1.add_subplot(111)
ax2 = fig2.add_subplot(111)

data_line = np.zeros(ngrid+1)

for spec in range(nspec):
    data_line = relative_error[:,spec]
    if( (np.mean(data_line) > 0.7E-3 and np.mean(data_line) < 1.0E10) or False ):
        ax1.plot(data_line, label=spec)
        ax2.plot(my_abn[:,spec], label=spec)
        ax2.plot(their_abn[:,spec], label=spec)


ax1.legend()
ax1.set_title("abundances relative error")
ax1.set_xlabel("x (grid point)")
ax1.set_ylabel("abundances relative error")
ax1.grid()
ax1.set_yscale("log")

ax2.legend()
ax2.set_title("abundances relative error")
ax2.set_xlabel("x (grid point)")
ax2.set_ylabel("abundances relative error")
ax2.grid()
ax2.set_yscale("log")

fig1.tight_layout()
fig2.tight_layout()

plot_name1 = "error_abundances" + tag + ".png"
plot_name2 = "both_abundances" + tag + ".png"



# Save the plot in pdf format

fig1.savefig(plot_name1, bbox_inches='tight')
fig2.savefig(plot_name2, bbox_inches='tight')


print "Plots saved as "
print "   " + plot_name1
print "   " + plot_name2
print " "
