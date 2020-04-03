#  Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
#
#  Developed by: Frederik De Ceuster - University College London & KU Leuven
#  _________________________________________________________________________


import numpy as np

from healpy   import pixelfunc
from magritte import Model
from setup    import Setup
from rays     import rayVectors

def sphericalXDscalar (fr, ndirs, f):
    '''
    Copy 1D model scalar data over shell in XD model
    '''
    if (ndirs == 0):
        f.append (fr)
    else:
        for _ in range (ndirs):
            f.append (fr)


#def sphericalXDarray (ar, ndirs, a):
#    '''
#    Copy 1D model scalar data over shell in 2D model
#    '''
#    if (ndirs == 0):
#
#        a.append (ar)
#    else:
#        for _ in range (ndirs):
#            f.append (fr)


def sphericalXDvector(Vr, dimension, ndirs, V):
    '''
    Copy 1D model scalar data over shell in XD model
    '''
    if (ndirs == 0):
        V.append(np.array([Vr, Vr, Vr]))
    else:
        # Create ray directions
        (Rx, Ry, Rz) = rayVectors(dimension, ndirs, randomize=True)
        # Extend vectors
        for r in range (ndirs):
            V.append(Vr*np.array([Rx[r], Ry[r], Rz[r]]))


def mapToXD (model1D, dimension, nrays, cellsInShell):
    """
    Maps a 1D model to the spherically symmetric XD equivalent
    """
    # Create model and setup objects
    model = Model ()
    setup = Setup (dimension = dimension)

    position    = []
    velocity    = []
    abundance   = []
    temperature = []
    turbulence  = []

    # Add shells
    for s in range (model1D.parameters.ncells()):
        ndirs = len (cellsInShell[s])
        sphericalXDvector(model1D.geometry.cells.position[s][0], dimension, ndirs, position)
        sphericalXDvector(model1D.geometry.cells.velocity[s][0], dimension, ndirs, velocity)
        sphericalXDscalar(model1D.chemistry.species.abundance[s],           ndirs, abundance)
        sphericalXDscalar(model1D.thermodynamics.temperature.gas[s],        ndirs, temperature)
        sphericalXDscalar(model1D.thermodynamics.turbulence.vturb2[s],      ndirs, turbulence)

    model.geometry.cells.position          = position
    model.geometry.cells.velocity          = velocity
    model.chemistry.species.abundance      = abundance
    model.thermodynamics.temperature.gas   = temperature
    model.thermodynamics.turbulence.vturb2 = turbulence

    # Extract number of cells
    ncells = 0
    for shell in cellsInShell:
        ncells += len (shell)
    # Extract boundary
    model.geometry.boundary.boundary2cell_nr = cellsInShell[-1]
    # Add rays
    model.geometry.rays = setup.rays (nrays=nrays, cells=model.geometry.cells)
    # Set ncells
    model.parameters.set_ncells (ncells)
    # Extract neighbors
    model.geometry.cells = setup.neighborLists (cells=model.geometry.cells)
    # Add linedata
    model.lines = model1D.lines
    # Add chemical species
    model.chemistry.species.sym = model1D.chemistry.species.sym
    # Set other parameters
    model.parameters.set_nrays  (nrays)
    model.parameters.set_nspecs (model1D.parameters.nspecs())
    model.parameters.set_nlspecs(model1D.parameters.nlspecs())
    model.parameters.set_nquads (model1D.parameters.nquads())
    # Done
    return model
