import yt
import matplotlib.pyplot as plt
import numpy as np


# Make a box containing the Magritte input


# if (len(sys.argv)>1):
#     name       =  str(sys.argv[1])
#     input_file = "../input/{}.txt".format(name)
# else:
#     print("ERROR : No input file given !\n")
#     print("        First argument is input file, second argument is box_size." )
#
# if (len(sys.argv)>2):
#     box_size = int(sys.argv[2])
# else:
#     print("ERROR : No input file given !\n")
#     print("        First argument is input file, second argument is box_size." )



box_size = 10

Magritte_input = '../input/CC_112_b_0.001.txt'

x,y,z, vx,vy,vz, density = np.loadtxt(Magritte_input, unpack=True)

NGRID = np.shape(density)[0]

temperature_a = np.loadtxt('temperature_a.txt')
temperature_b = np.loadtxt('temperature_b.txt')

x_min = min(x)
x_max = max(x)
y_min = min(y)
y_max = max(y)
z_min = min(z)
z_max = max(z)

box_density = np.zeros( (box_size, box_size, box_size) )
box_tempe_a = np.zeros( (box_size, box_size, box_size) )
box_tempe_b = np.zeros( (box_size, box_size, box_size) )
box_npoints = np.zeros( (box_size, box_size, box_size) )

for gridp in range(NGRID):
    n_x = int( (x[gridp] - x_min) / (x_max-x_min) * (box_size-1) )
    n_y = int( (y[gridp] - y_min) / (y_max-y_min) * (box_size-1) )
    n_z = int( (z[gridp] - z_min) / (z_max-z_min) * (box_size-1) )

    box_density[n_x, n_y, n_z] = box_density[n_x, n_y, n_z]*box_npoints[n_x, n_y, n_z] + density[gridp]
    box_tempe_a[n_x, n_y, n_z] = box_tempe_a[n_x, n_y, n_z]*box_npoints[n_x, n_y, n_z] + temperature_a[gridp]
    box_tempe_b[n_x, n_y, n_z] = box_tempe_b[n_x, n_y, n_z]*box_npoints[n_x, n_y, n_z] + temperature_b[gridp]

    box_npoints[n_x, n_y, n_z] = box_npoints[n_x, n_y, n_z] + 1
    box_density[n_x, n_y, n_z] = box_density[n_x, n_y, n_z] / box_npoints[n_x, n_y, n_z]
    box_tempe_a[n_x, n_y, n_z] = box_tempe_a[n_x, n_y, n_z] / box_npoints[n_x, n_y, n_z]
    box_tempe_b[n_x, n_y, n_z] = box_tempe_b[n_x, n_y, n_z] / box_npoints[n_x, n_y, n_z]

for i in range(box_size):
    fig = plt.figure()
    ax1=fig.add_subplot(131)
    ax2=fig.add_subplot(132)
    ax3=fig.add_subplot(133)

    array = box_density[:,i,:]
    ax1.imshow(array, cmap='hot', interpolation='nearest')
    # ax1.colormap()

    array = box_tempe_a[:,i,:]
    ax2.imshow(array, cmap='hot', interpolation='nearest')
    # ax2.colormap()

    array = box_tempe_a[:,i,:]
    ax3.imshow(array, cmap='hot', interpolation='nearest')

    plot_name = 'TB_temp_{}.png'.format(i)
    fig.savefig(plot_name, bbox_inches='tight')
