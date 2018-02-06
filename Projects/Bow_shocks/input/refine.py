import numpy as np
import sys



# Doubles the number of cells by interpolation



file_name = str(sys.argv[1])

input_data = np.loadtxt(file_name+".txt")

ncells     = np.shape(input_data)[0]
ncells_new = 2*ncells-1


x_new       = np.zeros(ncells_new)
y_new       = np.zeros(ncells_new)
z_new       = np.zeros(ncells_new)

vx_new      = np.zeros(ncells_new)
vy_new      = np.zeros(ncells_new)
vz_new      = np.zeros(ncells_new)

density_new = np.zeros(ncells_new)


for i in range(ncells):

    x_new[2*i]       = input_data[i,0]
    y_new[2*i]       = input_data[i,1]
    z_new[2*i]       = input_data[i,2]

    vx_new[2*i]      = input_data[i,3]
    vy_new[2*i]      = input_data[i,4]
    vz_new[2*i]      = input_data[i,5]

    density_new[2*i] = input_data[i,6]


for i in range(1,ncells):

    x_new[2*i-1]       = (input_data[i,0]+input_data[i-1,0])/2.0
    y_new[2*i-1]       = (input_data[i,1]+input_data[i-1,1])/2.0
    z_new[2*i-1]       = (input_data[i,2]+input_data[i-1,2])/2.0

    vx_new[2*i-1]      = (input_data[i,3]+input_data[i-1,3])/2.0
    vy_new[2*i-1]      = (input_data[i,4]+input_data[i-1,4])/2.0
    vz_new[2*i-1]      = (input_data[i,5]+input_data[i-1,5])/2.0

    density_new[2*i-1] = (input_data[i,6]+input_data[i-1,6])/2.0

input_data_new = np.stack((x_new, y_new, z_new, vx_new, vy_new, vz_new, density_new), axis=1)
input_filename_new = file_name + "_refined.txt"
np.savetxt(input_filename_new, input_data_new, fmt='%lE\t%lE\t%lE\t%lE\t%lE\t%lE\t%lE')
