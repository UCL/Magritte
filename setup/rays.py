# Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
#
# Developed by: Frederik De Ceuster - University College London & KU Leuven
# _________________________________________________________________________


import numpy as np

from healpy import pixelfunc


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


def rayVectors (dimension, nrays):
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
        Rx = [np.cos((2.0*np.pi*r)/nrays) for r in range(nrays)]
        Ry = [np.sin((2.0*np.pi*r)/nrays) for r in range(nrays)]
        Rz = [0.0                         for _ in range(nrays)]
    elif (dimension == 3):
        Rx = pixelfunc.pix2vec(nSides(nrays), range(nrays))[0]
        Ry = pixelfunc.pix2vec(nSides(nrays), range(nrays))[1]
        Rz = pixelfunc.pix2vec(nSides(nrays), range(nrays))[2]
    else:
        raise ValueError ('dimension shound be 1, 2, or 3.')
    # Done
    return (Rx, Ry, Rz)
