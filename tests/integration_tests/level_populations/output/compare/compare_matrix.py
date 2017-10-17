import numpy as np
import matplotlib.pyplot as plt
import sys



name = str(sys.argv[1])

if(len(sys.argv) > 2):
    tag  = "_" + str(sys.argv[2])
else:
    tag = ""


file_name = "../files/" + name + tag + ".txt"

print file_name

my_data = np.loadtxt(file_name)

ngrid  = np.shape(my_data)[0]
nindex = np.shape(my_data)[1]


file_name = "output_3D-PDR/" + name + tag + "_3D-PDR.txt"

their_data = np.loadtxt(file_name)



error          = my_data - their_data
relative_error = 2.0*abs(error)/abs(my_data+their_data)





# Make the plots

print " "
print "Plotting " + file_name


fig1 = plt.figure()
# fig2 = plt.figure()

# ax1 = fig1.add_subplot(111)
# ax2 = fig2.add_subplot(111)

plt.imshow(relative_error)
plt.colorbar()

plot_name1 = "error_" + name + tag + ".png"
fig1.savefig(plot_name1, bbox_inches='tight')

# data_line = np.zeros(ngrid)
#
#
# for index in range(nindex):
#
#     data_line = relative_error[:,index]
#
#     if(np.mean(data_line) > 0.1E-10 or True):
#
#         ax1.plot(data_line, label=index)
#
#
#
# # ax1.legend()
# ax1.set_title(name + tag + " error")
# ax1.set_xlabel("n (grid point nr)")
# ax1.set_ylabel(name + " error")
# ax1.grid()
# ax1.set_yscale("log")
#
# fig1.tight_layout()
#
#
# # ax2.legend()
# # ax2.set_title("both " + name + tag)
# # ax2.set_xlabel("n (grid point nr)")
# # ax2.set_ylabel(name + " error")
# # ax2.grid()
# # ax2.set_yscale("log")
# #
# # fig2.tight_layout()
#
#
# plot_name1 = "error_" + name + tag + ".png"
# plot_name2 = "both_" + name + tag + ".png"
#
#
# # Save the plot
#

# # fig2.savefig(plot_name2, bbox_inches='tight')
#
#
# print "Plots saved as "
# print "   " + plot_name1
# # print "   " + plot_name2
# print " "
