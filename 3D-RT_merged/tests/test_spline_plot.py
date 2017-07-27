import numpy as np
import matplotlib.pyplot as plt

xa, ya, d2y = np.loadtxt("test_spline_table.txt", unpack=True)
x , y       = np.loadtxt("test_spline_func.txt", unpack=True)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(xa, ya, 'ro')
ax.plot(x , y )

# plt.show()
fig.tight_layout()
fig.savefig("test_spline_plot.pdf", bbox_inches='tight')