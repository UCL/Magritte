import numpy as np
import matplotlib.pyplot as plt
import sys



name = str(sys.argv[1])

file_name = "../" + name + ".txt"

data = np.loadtxt(file_name)



# Make the plots

print " "
print "Plotting " + file_name

fig = plt.figure()

ax1 = fig.add_subplot(111)


# for spec in range(nspec):
#     for point in range(ngrid):
#         abundance[point] = abundances_data[spec][point]
#
#     if( max(abundance) > 1.0E-10 ):
#         ax1.plot(abundance, label=species_name[spec])

ax1.plot(data)

ax1.set_title(name)
ax1.set_xlabel("x (grid point)")
ax1.set_ylabel(name)
# ax1.set_yscale("log")

fig.tight_layout()

plot_name = name + ".png"



# Save the plot in pdf format

fig.savefig(plot_name, bbox_inches='tight')

print "Plot saved as " + plot_name
print " "
