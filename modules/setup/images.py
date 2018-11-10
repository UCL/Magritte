import numpy as np


sRx2pRy2 = np.sqrt(Rx**2 + Ry**2)

Ix =  Ry / sRx2pRy2
Iy = -Rx / sRx2pRy2
Iz =  0.0  

Jx = Rx * Rz / sRx2pRy2
Jy = Ry * Rz / sRx2pRy2
Jz = -sRx2pRy2
