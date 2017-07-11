import numpy as np
import matplotlib.pyplot as plt

xa, ya = np.loadtxt("X_lambda.txt", unpack=True)
x , y  = np.loadtxt("X_lambda_spline.txt", unpack=True)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(xa, ya, 'ro')
ax.plot(x , y )

# plt.show()
fig.tight_layout()
fig.savefig("test_rate_calculations_radfield_plot.pdf", bbox_inches='tight')