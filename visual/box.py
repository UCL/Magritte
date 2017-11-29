import yt
import matplotlib.pyplot as plt
import numpy as np


# Make a box_size x box_size x box_size box containing the Magritte input


if (len(sys.argv)>1):
    input_file = str(sys.argv[1])
else:
    print("ERROR : No input file given !\n")
    print("        First argument is the input file, second argument is the box size." )

if (len(sys.argv)>2):
    box_size = str(sys.argv[2])
else:
    print("ERROR : No input file given !\n")
    print("        First argument is the input file, second argument is the box size." )


box_size = 3

input_file = '../input/CC_112_b.txt'

x, y, z, density = np.loadtxt(input_file, unpack=True)

NGRID = np.shape(density)[0]

x_min = min(x)
x_max = max(x)
y_min = min(y)
y_max = max(y)
z_min = min(z)
z_max = max(z)

box_density = np.zeros( (box_size, box_size, box_size) )
box_npoints = np.zeros( (box_size, box_size, box_size) )

for gridp in range(NGRID):
    n_x = int( (x[gridp] - x_min) / (x_max-x_min) * (box_size-1) )
    n_y = int( (y[gridp] - y_min) / (y_max-y_min) * (box_size-1) )
    n_z = int( (z[gridp] - z_min) / (z_max-z_min) * (box_size-1) )

    box_density[n_x, n_y, n_z] = box_density[n_x, n_y, n_z]*box_npoints[n_x, n_y, n_z] + density[gridp]
    box_npoints[n_x, n_y, n_z] = box_npoints[n_x, n_y, n_z] + 1
    box_density[n_x, n_y, n_z] = box_density[n_x, n_y, n_z] / box_npoints[n_x, n_y, n_z]


# plt.hist( lin_npoints, bins=100 )
ax1.plot(lin_npoints)


ax2 = fig.add_subplot(212)

ax2.plot(lin_density)
