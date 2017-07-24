import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



# ----------------------------------------
# Create a 1D grid to test Ray Tracing
# ----------------------------------------



def randrange(n, vmin, vmax):
    '''
    Helper function to make an array of random numbers having shape (n, )
    with each number distributed Uniform(vmin, vmax).
    '''
    return (vmax - vmin)*np.random.rand(n) + vmin



ngrid = 50 #int(input("How many grid points?"))

np.random.seed(9001)

x = 0.523797/ngrid*np.array(range(ngrid))
y = 0.783917/ngrid*np.array(range(ngrid))
z = 0.333333/ngrid*np.array(range(ngrid))

vx = np.zeros(ngrid)
vy = np.zeros(ngrid)
vz = np.zeros(ngrid)

data = np.stack((x, y, z, vx, vy, vz), axis=1)
np.savetxt('grid.txt', data, fmt='%lf\t%lf\t%lf\t%lf\t%lf\t%lf')

print("Grid (1D) created with", ngrid, "grid points!")

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
