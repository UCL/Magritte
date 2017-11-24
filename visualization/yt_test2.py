import yt
import numpy as np

def linramp(vals, minval, maxval):
    return (vals - vals.min())/(vals.max() - vals.min())


# Make a box containing the Magritte input
box_size = 2
Magritte_input = '../input/CC_112_b.txt'
x, y, z, density = np.loadtxt(Magritte_input, unpack=True)

NGRID = np.shape(density)[0]

x_min = min(x)
x_max = max(x)
y_min = min(y)
y_max = max(y)
z_min = min(z)
z_max = max(z)

box_density = np.zeros((box_size, box_size, box_size))
box_npoints = np.zeros((box_size, box_size, box_size))

for gridp in range(NGRID):
    n_x = int((x[gridp] - x_min) / (x_max-x_min) * (box_size-1))
    n_y = int((y[gridp] - y_min) / (y_max-y_min) * (box_size-1))
    n_z = int((z[gridp] - z_min) / (z_max-z_min) * (box_size-1))

    box_density[n_x, n_y, n_z] = box_density[n_x, n_y, n_z] * box_npoints[n_x, n_y, n_z] + density[gridp]
    box_npoints[n_x, n_y, n_z] = box_npoints[n_x, n_y, n_z] + 1
    box_density[n_x, n_y, n_z] = box_density[n_x, n_y, n_z] / box_npoints[n_x, n_y, n_z]

data = dict(density = (box_density, "g/cm**3"))
bbox = np.array([[-1.5, 1.5], [-1.5, 1.5], [-1.5, 1.5]])
ds = yt.load_uniform_grid(data, box_density.shape, length_unit="Mpc", bbox=bbox, nprocs=2)

volume_render = True

if volume_render:
    #Create a volume rendering
    sc = yt.create_scene(ds, field=('gas', 'density'))

    # Modify the transfer function

    # First get the render source, in this case the entire domain, with field ('gas','density')
    render_source = sc.get_source()

    # Clear the transfer function
    render_source.transfer_function.clear()

    # Map a range of density values (in log space) to the Reds_r colormap
    render_source.transfer_function.map_to_colormap(
        np.min(box_density), np.max(box_density),
        scale=.01, colormap='afmhot')#, scale_func = linramp )

    sc.save('new_tf.png')
else:
    slc = yt.SlicePlot(ds, "z", ["density"])
    slc.set_cmap("density", "Blues")
    slc.annotate_grids(cmap=None)
    slc.save()
