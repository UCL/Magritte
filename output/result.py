import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

result = np.loadtxt("result.txt", skiprows=2)


# def func(x,A,B)
#   return 

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(result)
plt.show()