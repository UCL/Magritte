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

my_rates = np.loadtxt(file_name)

ngrid = np.shape(my_rates)[0]
nreac = np.shape(my_rates)[1]


file_name = "output_3D-PDR/" + name + "_rates" + tag + "_3D-PDR.txt"

their_rates = np.loadtxt(file_name)


error          = my_rates - their_rates
relative_error = 2.0*abs(error)/(my_rates+their_rates)

print " "
print "Do the reaction numbers agree?"

# for item in error[0][:]
#     if(item != 0):
#         print "Reaction " + item + "has a wrong partner!"


# Make the plots

print " "
print "Plotting " + file_name

fig = plt.figure()

ax1 = fig.add_subplot(111)

data_line = np.zeros(ngrid+1)

for reac in range(nreac):
    data_line = relative_error[:,reac]
    if( (np.mean(data_line) > 0.5E-1 and np.mean(data_line) < 1.0E10) or True ):
        ax1.plot(data_line, label=int(my_rates[0,reac]))
        # ax1.plot(my_rates[:,reac], label=int(my_rates[0,reac]))
        # ax1.plot(their_rates[:,reac], label=int(my_rates[0,reac]))
    if(False):
        ax1.plot(my_rates[:,reac], label=int(my_rates[0,reac]))
        ax1.plot(their_rates2[:,reac], label=int(their_rates[0,reac]))

# ax1.legend()
ax1.set_title(name + " relative error")
ax1.set_xlabel("x (grid point)")
ax1.set_ylabel(name + " relative error")
ax1.set_yscale("log")

fig.tight_layout()

plot_name = "error_" + name + ".png"


# Save the plot in pdf format

fig.savefig(plot_name, bbox_inches='tight')

print "Plot saved as " + plot_name
print " "
