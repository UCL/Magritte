import numpy as np
import matplotlib.pyplot as plt


# DEFINITIONS

###################################################################################################

def crop_data( xl, xh, yl, yh ):
    # Crop the data to be in the range [xl, xh] and [yl, yh]
    cropped_IDs = []
    for gridp in range(NCELLS):
        if ( (x[gridp] > xl) and (x[gridp] < xh) and (y[gridp] > yl) and (y[gridp] < yh) ):
            cropped_IDs.append( int(ID[gridp]) )
    return cropped_IDs

###################################################################################################




# Read data

ID, x,y,z, vx,vy,vz, density = np.loadtxt(fname="grid.txt",unpack=True)

NCELLS = len(density)


# Crop data

xl_bound = +0.0E00
xh_bound = +8.0E16

yl_bound = -1.6E17
yh_bound = +6.0E16

new_IDs = crop_data(xl_bound, xh_bound, yl_bound, yh_bound)

x_c = [x[gridp] for gridp in new_IDs]
y_c = [y[gridp] for gridp in new_IDs]
z_c = [z[gridp] for gridp in new_IDs]

density_c = [density[gridp] for gridp in new_IDs]


# Box data

x_min = min(x_c)
x_max = max(x_c)
y_min = min(y_c)
y_max = max(y_c)
# z_min = min(z_c)
# z_max = max(z_c)

box_size = 84

box_density = np.zeros( (box_size, box_size) )
box_npoints = np.zeros( (box_size, box_size) )

for gridp in new_IDs:
    n_x = int( (x[gridp] - x_min) / (x_max-x_min) * (box_size-1) )
    n_y = int( (y[gridp] - y_min) / (y_max-y_min) * (box_size-1) )

    box_density[n_x, n_y] = box_density[n_x, n_y]*box_npoints[n_x, n_y] + density[gridp]
    box_npoints[n_x, n_y] = box_npoints[n_x, n_y] + 1
    box_density[n_x, n_y] = box_density[n_x, n_y] / box_npoints[n_x, n_y]



# Plot data

fig1 = plt.figure()
fig2 = plt.figure()

sub1 = fig1.add_subplot(211)
sub2 = fig1.add_subplot(212)

sta1 = fig2.add_subplot(211)
sta2 = fig2.add_subplot(212)

log_box_density = np.log(box_density)
log_box_npoints = np.log(box_npoints)

sub1.imshow( X=log_box_density, aspect=(x_max-x_min)/(y_max-y_min), \
             cmap='hot', interpolation='gaussian' )
sub2.imshow( X=box_npoints, aspect=(x_max-x_min)/(y_max-y_min),     \
             cmap='hot', interpolation='nearest' )

sta1.hist(np.log(density), bins=100)
sta2.hist(np.log(density_c), bins=100)

sub1.axis('off')
sub2.axis('off')

fig1.savefig("grid.pdf", bbox_inches='tight')
fig2.savefig("grid_stats.pdf", bbox_inches='tight')


# Reduce data


# Find the mininal non-zero increment in x and y
# This should define boxes with at most one cell

x_sorted = np.sort(x)
y_sorted = np.sort(y)

dx = [x_sorted[i]-x_sorted[i-1] for i in range(1,len(x))]
dy = [y_sorted[i]-y_sorted[i-1] for i in range(1,len(y))]

min_x = max(x)
min_y = max(y)

for i in range(len(x)-1):
    if (dx[i] < min_x) and (dx[i] > 0.0):
        min_x = dx[i]

for i in range(len(y)-1):
    if (dy[i] < min_y) and (dy[i] > 0.0):
        min_y = dy[i]

dx_min = 2.0*min_x
dy_min = 2.0*min_y

NSTEPS_x = int( (x_max-x_min)/dx_min )
NSTEPS_y = int( (y_max-y_min)/dy_min )

print(NSTEPS_x, NSTEPS_y)


# Box the data in this minimal grid

box_density = np.zeros( (NSTEPS_x, NSTEPS_y) )
box_npoints = np.zeros( (NSTEPS_x, NSTEPS_y) )

for i in new_IDs:
    n_x = int( (x[i] - x_min) / (x_max-x_min) * (NSTEPS_x-1) )
    n_y = int( (y[i] - y_min) / (y_max-y_min) * (NSTEPS_y-1) )

    box_density[n_x, n_y] = box_density[n_x, n_y]*box_npoints[n_x, n_y] + density[gridp]
    box_npoints[n_x, n_y] = box_npoints[n_x, n_y] + 1
    box_density[n_x, n_y] = box_density[n_x, n_y] / box_npoints[n_x, n_y]


# Remove neighbors if theiy are similar to the center

pre = 1.0E-3

delete = []

for ix in range(1, NSTEPS_x-1, 2):
    for iy in range(1, NSTEPS_y-1, 2):

        rho = box_density[ix,iy]

        c1 = (box_density[ix-1,iy-1]/rho < pre) and (box_density[ix-1,iy]/rho < pre)
        c2 = (box_density[ix-1,iy+1]/rho < pre) and (box_density[ix,iy+1]/rho < pre)
        c3 = (box_density[ix,iy-1]/rho < pre) and (box_density[ix+1,iy+1]/rho < pre)
        c4 = (box_density[ix+1,iy]/rho < pre) and (box_density[ix+1,iy-1]/rho < pre)

        if c1 and c2 and c3 and c4:
            delete.append([ix-1,iy-1])
            delete.append([ix-1,iy])
            delete.append([ix-1,iy+1])
            delete.append([ix,iy+1])
            delete.append([ix,iy-1])
            delete.append([ix+1,iy+1])
            delete.append([ix+1,iy])
            delete.append([ix+1,iy-1])

print len(new_IDs)

fin_IDs = new_IDs

for i in range(len(new_IDs)):
    n_x = int( (x[new_IDs[i]] - x_min) / (x_max-x_min) * (NSTEPS_x-1) )
    n_y = int( (y[new_IDs[i]] - y_min) / (y_max-y_min) * (NSTEPS_y-1) )

    print [n_x, n_y] in delete

    # if [n_x, n_y] in delete:
    #     del fin_IDs[i]

    # for i in range(len(delete)):
    #     if (delete[i] == [n_x, n_y]):
    #         del fin_IDs[i]


print len(fin_IDs)


fig1 = plt.figure()

sub1 = fig1.add_subplot(211)
sub2 = fig1.add_subplot(212)

log_box_density = np.log(box_density)
log_box_npoints = np.log(box_npoints)

sub1.imshow(X=log_box_density, cmap='hot', interpolation='gaussian')
sub2.imshow(X=box_npoints, cmap='hot', interpolation='nearest')

sub1.axis('off')
sub2.axis('off')

fig1.savefig("grid.pdf", bbox_inches='tight')
