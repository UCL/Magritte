import yt
import numpy as np
import sys

# def linramp(vals, minval, maxval):
#     return .05*((vals - vals.min())/(vals.max() - vals.min()))**1.5

def linramp(vals, minval, maxval):
    return .05*((vals - vals.min())/(vals.max() - vals.min()))**1.7



if (len(sys.argv)>1):
    name       =  str(sys.argv[1])
    input_file = "../input/{}.txt".format(name)
else:
    print("ERROR : No input file given !\n")
    print("        First argument is input file, second argument is box_size." )

if (len(sys.argv)>2):
    box_size = int(sys.argv[2])
else:
    print("ERROR : No input file given !\n")
    print("        First argument is input file, second argument is box_size." )


# Make a box containing the Magritte input
# box_size = 21
# box_size = 110

# Magritte_input = 'grid.txt'
# Magritte_input = '../input/CC_112_b.txt'

x,y,z, vx,vy,vz,  density = np.loadtxt(input_file, unpack=True)
# x,y,z,  density = np.loadtxt(Magritte_input, unpack=True)

NCELLS = np.shape(density)[0]

x_min = min(x)
x_max = max(x)
y_min = min(y)
y_max = max(y)
z_min = min(z)
z_max = max(z)

box_density = np.zeros((box_size, box_size, box_size))
box_npoints = np.zeros((box_size, box_size, box_size))

for gridp in range(NCELLS):
    n_x = int((x[gridp] - x_min) / (x_max-x_min) * (box_size-1))
    n_y = int((y[gridp] - y_min) / (y_max-y_min) * (box_size-1))
    n_z = int((z[gridp] - z_min) / (z_max-z_min) * (box_size-1))

    box_density[n_x, n_y, n_z] = box_density[n_x, n_y, n_z] * box_npoints[n_x, n_y, n_z] + density[gridp]
    box_npoints[n_x, n_y, n_z] = box_npoints[n_x, n_y, n_z] + 1
    box_density[n_x, n_y, n_z] = box_density[n_x, n_y, n_z] / box_npoints[n_x, n_y, n_z]



data = dict(density = (box_density, "g/cm**3"))
box_ax = 1.5
bbox = np.array([[-box_ax, box_ax], [-box_ax, box_ax], [-box_ax, box_ax]])
ds = yt.load_uniform_grid(data, box_density.shape, length_unit="Mpc", bbox=bbox, nprocs=20)



volume_render = True

# if volume_render:
#Create a volume rendering
sc = yt.create_scene(ds, field=('gas', 'density'))

# Modify the transfer function

# First get the render source, in this case the entire domain, with field ('gas','density')
render_source = sc.get_source()

# Clear the transfer function
render_source.transfer_function.clear()

# Map a range of density values (in log space) to the Reds_r colormap
render_source.transfer_function.map_to_colormap( np.min(box_density), 0.1*np.max(box_density), scale=0.1, colormap='binary', scale_func = linramp )


file_name = "{}.png".format(name)
sc.save(file_name)

# else:



slc_x = yt.SlicePlot(ds, "x", ["density"])
slc_x.set_cmap("density", "binary")
slc_x.annotate_grids(cmap=None)
slc_x.save(name)

slc_y = yt.SlicePlot(ds, "y", ["density"])
slc_y.set_cmap("density", "binary")
slc_y.annotate_grids(cmap=None)
slc_y.save(name)

slc_z = yt.SlicePlot(ds, "z", ["density"])
slc_z.set_cmap("density", "binary")
slc_z.annotate_grids(cmap=None)
slc_z.save(name)
