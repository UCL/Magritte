from mpl_toolkits.mplot3d import Axes3D

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
fig.savefig("test_X_lambda.pdf", bbox_inches='tight')


log10column_COa, log10column_H2a, log10shield_COa = np.loadtxt("self_shielding_CO_table.txt", unpack=True)
log10column_CO , log10column_H2 , log10shield_CO  = np.loadtxt("self_shielding_CO_spline.txt", unpack=True)

fig2 = plt.figure()
ax2 = fig2.gca(projection='3d')
ax2.scatter(log10column_COa, log10column_H2a, log10shield_COa)
ax2.plot_trisurf(log10column_CO, log10column_H2, log10shield_CO )

# plt.show()
# fig2.tight_layout()
# fig2.savefig("test_self_shielding_CO.pdf", bbox_inches='tight')