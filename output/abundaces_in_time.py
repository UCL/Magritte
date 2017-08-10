import numpy as np
import matplotlib.pyplot as plt

abundances = np.loadtxt("abundances_in_time.txt")

nout  = np.shape(abundances)[0]
nspec = np.shape(abundances)[1]

print "nout is  ", nout
print "nspec is ", nspec


fig = plt.figure()
ax1 = fig.add_subplot(111)

abn = np.zeros(nout)


for spec in range(nspec):

    for time in range(nout):
         abn[time] = abundances[time][spec]

    ax1.plot(abn, label=spec)


ax1.legend()
ax1.set_xlabel("time")
ax1.set_ylabel("abundance")
ax1.set_yscale("log")

plt.show()
