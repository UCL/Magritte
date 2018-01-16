import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys



# -----------------------------------
# Create a list of temperatures in 1D
# -----------------------------------


# Get desired number of cells form argument

ncells = int(sys.argv[1])



def temperature1(x,alpha):
    return np.power(x,alpha)


temperature = 10

data = np.stack(temperature, axis=1)
filename = "files/temperature_" + str(ncells) + ".txt"
np.savetxt(filename, data, fmt='%lE')

print("Grid (1D) created with", ncells, "grid points!")

# Plot the resulting grid

#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.scatter(x, y, z)
#ax.set_xlabel("x")
#ax.set_ylabel("y")
#ax.set_zlabel("z")

#fig.tight_layout()

#plt.show()
#fig.savefig("plots.pdf", bbox_inches='tight')
