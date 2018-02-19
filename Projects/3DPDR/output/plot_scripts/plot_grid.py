import matplotlib.pyplot as plt
import numpy as np
import sys
from matplotlib.colors import LogNorm

# Check whether the date stamp of the datafile is given
if (len(sys.argv) > 1):
    date_stamp = str(sys.argv[1])
else:
    print "ERROR : No date stamp given !\n"
    print "Please try again and give the date stamp of the output file you want to plot\n"


# Check the tag of the data that is to be plotted
if (len(sys.argv)>2):
    tag = "_" + str(sys.argv[2])
else:
    tag = ""


# Read data
file_name = "../files/{}_output/grid{}.txt".format(date_stamp, tag)

ID, x,y,z, vx,vy,vz, density = np.loadtxt(file_name, unpack = True)

NCELLS = len(ID)

print "number of cells {}".format(NCELLS)


# Box data
x_min = min(x)
x_max = max(x)
y_min = min(y)
y_max = max(y)
z_min = min(z)
z_max = max(z)

box_size = 43

box_density = np.zeros( (box_size, box_size) )
box_npoints = np.zeros( (box_size, box_size) )

for o in range(NCELLS):
    n_x = int( (x[o] - x_min) / (x_max-x_min) * (box_size-1) )
    n_y = int( (y[o] - y_min) / (y_max-y_min) * (box_size-1) )

    box_density[n_x, n_y] = box_density[n_x, n_y]*box_npoints[n_x, n_y] + density[o]
    box_npoints[n_x, n_y] = box_npoints[n_x, n_y] + 1
    box_density[n_x, n_y] = box_density[n_x, n_y] / box_npoints[n_x, n_y]


# Plot data
fig1 = plt.figure()
fig2 = plt.figure()

sub1 = fig1.add_subplot(211)
sub2 = fig1.add_subplot(212)

sta1 = fig2.add_subplot(211)
sta2 = fig2.add_subplot(212)

density_min = 1.0E-24
density_max = 1.0E-18



log_box_density = np.log(box_density+1.0E-99)
log_box_npoints = np.log(box_npoints)

im1 = sub1.imshow( box_density, aspect=(x_max-x_min)/(y_max-y_min), \
                   cmap='hot', interpolation='gaussian',            \
                   norm=LogNorm(vmin=density_min, vmax=density_max) )
im2 = sub2.imshow( box_npoints, aspect=(x_max-x_min)/(y_max-y_min), \
                   cmap='hot', interpolation='nearest' )

sta1.hist(np.log(density), bins=100)

# sub1.axis('off')
# sub2.axis('off')

sub1.set_title("Material density")
sub2.set_title("Cell density, total = "+str(NCELLS))

fig1.colorbar(im1, ax=sub1)
fig1.colorbar(im2, ax=sub2)


fig_name1 = "../files/{}_output/plots/grid{}.pdf".format(date_stamp, tag)
fig_name2 = "../files/{}_output/plots/grid_stats{}.pdf".format(date_stamp, tag)

fig1.savefig(fig_name1, bbox_inches='tight')
fig2.savefig(fig_name2, bbox_inches='tight')


#
# # Reduce data
#
#
# # Find the mininal non-zero increment in x and y
# # This should define boxes with at most one cell
#
# x_sorted = np.sort(x)
# y_sorted = np.sort(y)
#
# dx = [x_sorted[i]-x_sorted[i-1] for i in range(1,len(x))]
# dy = [y_sorted[i]-y_sorted[i-1] for i in range(1,len(y))]
#
# min_x = max(x)
# min_y = max(y)
#
# for i in range(len(x)-1):
#     if (dx[i] < min_x) and (dx[i] > 0.0):
#         min_x = dx[i]
#
# for i in range(len(y)-1):
#     if (dy[i] < min_y) and (dy[i] > 0.0):
#         min_y = dy[i]
#
# dx_min = 2.0*min_x
# dy_min = 2.0*min_y
#
# NSTEPS_x = int( (x_max-x_min)/dx_min )
# NSTEPS_y = int( (y_max-y_min)/dy_min )
#
# print(NSTEPS_x, NSTEPS_y)
#
#
# # Box the data in this minimal grid
#
# box_density = np.zeros( (NSTEPS_x, NSTEPS_y) )
# box_npoints = np.zeros( (NSTEPS_x, NSTEPS_y) )
#
# for i in range(NCELLS):
#     n_x = int( (x[i] - x_min) / (x_max-x_min) * (NSTEPS_x-1) )
#     n_y = int( (y[i] - y_min) / (y_max-y_min) * (NSTEPS_y-1) )
#
#     box_density[n_x, n_y] = box_density[n_x, n_y]*box_npoints[n_x, n_y] + density[o]
#     box_npoints[n_x, n_y] = box_npoints[n_x, n_y] + 1
#     box_density[n_x, n_y] = box_density[n_x, n_y] / box_npoints[n_x, n_y]
#
#
# # Remove neighbors if theiy are similar to the center
#
# pre = 1.0E-3
#
# delete = []
#
# for ix in range(1, NSTEPS_x-1, 2):
#     for iy in range(1, NSTEPS_y-1, 2):
#
#         rho = box_density[ix,iy]
#
#         c1 = (box_density[ix-1,iy-1]/rho < pre) and (box_density[ix-1,iy]/rho < pre)
#         c2 = (box_density[ix-1,iy+1]/rho < pre) and (box_density[ix,iy+1]/rho < pre)
#         c3 = (box_density[ix,iy-1]/rho < pre) and (box_density[ix+1,iy+1]/rho < pre)
#         c4 = (box_density[ix+1,iy]/rho < pre) and (box_density[ix+1,iy-1]/rho < pre)
#
#         if c1 and c2 and c3 and c4:
#             delete.append([ix-1,iy-1])
#             delete.append([ix-1,iy])
#             delete.append([ix-1,iy+1])
#             delete.append([ix,iy+1])
#             delete.append([ix,iy-1])
#             delete.append([ix+1,iy+1])
#             delete.append([ix+1,iy])
#             delete.append([ix+1,iy-1])
#
# print len(new_IDs)
#
# fin_IDs = new_IDs
#
# for i in range(len(new_IDs)):
#     n_x = int( (x[new_IDs[i]] - x_min) / (x_max-x_min) * (NSTEPS_x-1) )
#     n_y = int( (y[new_IDs[i]] - y_min) / (y_max-y_min) * (NSTEPS_y-1) )
#
#     print [n_x, n_y] in delete
#
#     # if [n_x, n_y] in delete:
#     #     del fin_IDs[i]
#
#     # for i in range(len(delete)):
#     #     if (delete[i] == [n_x, n_y]):
#     #         del fin_IDs[i]
#
#
# print len(fin_IDs)
#
#
# fig1 = plt.figure()
#
# sub1 = fig1.add_subplot(211)
# sub2 = fig1.add_subplot(212)
#
# log_box_density = np.log(box_density)
# log_box_npoints = np.log(box_npoints)
#
# sub1.imshow(log_box_density, cmap='hot', interpolation='gaussian')
# sub2.imshow(box_npoints, cmap='hot', interpolation='nearest')
#
# sub1.axis('off')
# sub2.axis('off')
#
# plot_name = "../files/" + date_stamp + "_output/plots/grid.pdf"
#
# fig1.savefig(plot_name, bbox_inches='tight')
