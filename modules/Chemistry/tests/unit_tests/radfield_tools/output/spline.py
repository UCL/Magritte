import numpy as np
import matplotlib.pyplot as plt

columnH2, sCO = np.loadtxt("spline.txt", unpack=True)

PDR = np.loadtxt("3D-PDR_spline.txt")

cpoints = np.array([18.0E0, 19.0E0, 20.0E0, 21.0E0, 22.0E0, 23.0E0])
spoints = np.array([0.000E+00, -8.539E-02, -1.451E-01, -4.559E-01, -1.303E+00, -3.883E+00])
cpoints = np.power(10.0,cpoints)




fig1 = plt.figure()

ax1 = fig1.add_subplot(111)


ax1.legend()
ax1.set_xlabel("columnH2")
ax1.set_ylabel("sCO")
# ax1.set_yscale("log")
ax1.set_xscale("log")

ax1.plot(columnH2, sCO)
ax1.plot(columnH2, PDR)
ax1.scatter(cpoints, spoints)

fig1.savefig("spline.png", bbox_inches='tight')
