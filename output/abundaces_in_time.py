import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("abundances_in_time.txt")

nout  = np.shape(data)[0]
nspecp1 = np.shape(data)[1]

print "nout is  ", nout
print "nspec is ", nspecp1


fig = plt.figure()
ax1 = fig.add_subplot(111)

abn  = np.zeros(nout)
time = np.zeros(nout)

for spec in range(1,nspecp1):

    for t in range(nout):
         abn[t]  = data[t][spec]
         time[t] = data[t][0]

    ax1.plot(abn, label=spec)


ax1.legend()
ax1.set_xlabel("time")
ax1.set_ylabel("abundance")
ax1.set_xscale("log")
ax1.set_yscale("log")

plt.show()
