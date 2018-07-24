import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("abundances_in_time.txt")
final = np.loadtxt("final_abundances.txt")

nout  = np.shape(data)[0]
nspecp1 = np.shape(data)[1]

print "nout is  ", nout
print "nspec is ", nspecp1


for spec in range(1,nspecp1):

    abn1 = data[nout-1][spec]
    abn2 = final[spec]

    print (abn1-abn2)/(abn1+abn2)





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
