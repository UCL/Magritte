# Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
#
# Developed by: Frederik De Ceuster - University College London & KU Leuven
# _________________________________________________________________________


import numpy as np

from healpy                  import pixelfunc
from scipy.spatial.transform import Rotation


def nRays (nsides):
    '''
    Number of rays corresponding to HEALPix's nsides.
    '''
    return 12*nsides**2


def nSides (nrays):
    '''
    Number of HEALPix's nsides corresponding to nrays.
    '''
    # Try computing nsides assuming it works
    nsides = int (np.sqrt (float(nrays) / 12.0))
    # Chack if nrays was HEALPix compatible
    if (nRays (nsides) != nrays):
        raise ValueError ('No HEALPix compatible nrays was given (nrays = 12*nsides**2).')
    # Done
    return nsides


def rayVectors (dimension, nrays, randomize=False):
    '''
    Devide 1, 2, or 3D sphere into 'nrays' rays.
    '''
    if   (dimension == 1):
        if (nrays != 2):
            raise ValueError ('In 1D, nrays should always be 2.')
        Rx = [-1.0, 1.0]
        Ry = [ 0.0, 0.0]
        Rz = [ 0.0, 0.0]
    elif (dimension == 2):
        if randomize:
            delta = np.random.uniform(0.0, 2.0*np.pi)
        Rx = [np.cos((2.0*np.pi*r)/nrays+delta) for r in range(nrays)]
        Ry = [np.sin((2.0*np.pi*r)/nrays+delta) for r in range(nrays)]
        Rz = [0.0                               for _ in range(nrays)]
    elif (dimension == 3):
        R = pixelfunc.pix2vec(nSides(nrays), range(nrays))
        if randomize:
            R = Rotation.random().apply(np.array(R).T).T
        (Rx, Ry, Rz) = (R[0], R[1], R[2])
    else:
        raise ValueError ('dimension shound be 1, 2, or 3.')
    # Done
    return (Rx, Ry, Rz)


def imageAxis (Rx, Ry, Rz):
    # Define help quantity
    sRx2pRy2 = np.sqrt(Rx**2 + Ry**2)
    if (sRx2pRy2 > 0.0):
        # Define unit vector along horizontal image axis
        Ix =  Ry / sRx2pRy2
        Iy = -Rx / sRx2pRy2
        Iz =  0.0
        # Define unit vector alonng vertical image axis
        Jx = Rx * Rz / sRx2pRy2
        Jy = Ry * Rz / sRx2pRy2
        Jz = -sRx2pRy2
        # Define image coordinates
        imageX = np.array([Ix, Iy, 0.])
        imageY = np.array([Jx, Jy, Jz])
    else:
        # Define image coordinates
        imageX = np.array([1., 0., 0.])
        imageY = np.array([0., 1., 0.])
    # Done
    return (imageX, imageY)
