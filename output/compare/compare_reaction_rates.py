import numpy as np
import matplotlib.pyplot as plt
import sys



name = str(sys.argv[1])


if(len(sys.argv) > 2):
    tag  = "_" + str(sys.argv[2])
else:
    tag = ""

if(name=="all"):
    name = "reaction"

print("Comparing")

file_name = "../files/" + name + "_rates" + tag + ".txt"

print(file_name)

my_rates = np.loadtxt(file_name)

ngrid = np.shape(my_rates)[0]
nreac = np.shape(my_rates)[1]


file_name = "output_3D-PDR/" + name + "_rates" + tag + "_3D-PDR.txt"

print(file_name)

their_rates = np.loadtxt(file_name)

rates             = their_rates[-2,:]
their_rates[-2,:] = their_rates[-1,:]
their_rates[-1,:] = their_rates[-2,:]


error          = my_rates - their_rates
relative_error = 2.0*abs(error)/(my_rates+their_rates)


# Make the plots

fig1 = plt.figure()
fig2 = plt.figure()

ax1 = fig1.add_subplot(111)
ax2 = fig2.add_subplot(111)

data_line = np.zeros(ngrid+1)

for reac in range(nreac):
    data_line = relative_error[:,reac]
    if( (np.mean(data_line[0:1]) > 1.0E-20 and np.mean(data_line) < 1.0E10) or False ):
        ax1.plot(data_line, label=int(reac))
        ax2.plot(my_rates[:,reac], label=int(reac))
        ax2.plot(their_rates[:,reac], label=int(reac))


# ax1.legend()
ax1.set_title(name + " relative error")
ax1.set_xlabel("n (grid point nr)")
ax1.set_ylabel(name + " relative error")
ax1.grid()
ax1.set_yscale("log")

ax2.legend()
ax2.set_title("both Magritte and 3D-PDR " + name + "s" + " at " + tag)
ax2.set_xlabel("n (grid point nr)")
ax2.set_ylabel(name)
ax2.grid()
ax2.set_yscale("log")

fig1.tight_layout()
fig2.tight_layout()

plot_name1 = "error_" + name + tag + ".png"
plot_name2 = "both_" + name + tag + ".png"


# Save the plot in pdf format

fig1.savefig(plot_name1, bbox_inches='tight')
fig2.savefig(plot_name2, bbox_inches='tight')

print("Plots saved as ")
print("   " + plot_name1)
print("   " + plot_name2)
print(" ")
