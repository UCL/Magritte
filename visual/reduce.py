import yt
import matplotlib.pyplot as plt
import numpy as np
import sys



if (len(sys.argv)>1):
    name       =  str(sys.argv[1])
    input_file = "{}.txt".format(name)
else:
    print("ERROR : No input file given !\n")
    print("        First argument is input file, second argument is reduction factor in ]0, 1[." )

if (len(sys.argv)>2):
    reduction = float(sys.argv[2])
else:
    print("ERROR : No input file given !\n")
    print("        First argument is input file, second argument is reduction factor in ]0, 1[." )


x, y, z, density = np.loadtxt(input_file, unpack=True)

NGRID = np.shape(density)[0]

x_new = []
y_new = []
z_new = []

density_new = []

np.random.seed(9001)

for gridp in range(NGRID):
    if ( reduction > np.random.rand() ):
        x_new.append(x[gridp])
        y_new.append(y[gridp])
        z_new.append(z[gridp])

        density_new.append(density[gridp])


NGRID_NEW = len(x_new)

print("New grid size is :" + str(NGRID_NEW))



vx = np.zeros(NGRID_NEW)
vy = np.zeros(NGRID_NEW)
vz = np.zeros(NGRID_NEW)

file_name = "{}_{}.txt".format(name, reduction)

data = np.stack((x_new, y_new, z_new, vx, vy, vz, density_new), axis=1)
np.savetxt(file_name, data, fmt='%lE\t%lE\t%lE\t%lE\t%lE\t%lE\t%lE')


plt.show()
