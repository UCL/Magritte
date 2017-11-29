import yt
import matplotlib.pyplot as plt
import numpy as np


# Make a box containing the Magritte input

box_size = 3

Magritte_input = '../input/CC_112_b.txt'

x, y, z, density = np.loadtxt(Magritte_input, unpack=True)

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

for i in range(box_size):
    fig = plt.figure()
    ax=fig.add_subplot(111)
    array = box_density[i,:,:]
    ax.imshow(array, cmap='hot', interpolation='nearest')
    plot_name = 'TB_x_{}.png'.format(i)
    fig.savefig(plot_name, bbox_inches='tight')
