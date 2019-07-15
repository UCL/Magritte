#  Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
#
#  Developed by: Frederik De Ceuster - University College London & KU Leuven
#  _________________________________________________________________________


import numpy as np

from healpy   import pixelfunc
from magritte import Model, Long1
from setup    import Setup
from rays     import rayVectors

def sphericalXDscalar (fr, ndirs, f):
    '''
    Copy 1D model scalar data over shell in 2D model
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


def sphericalXDvector(Vr, dimension, ndirs, Vx, Vy, Vz):
    '''
    Copy 1D model scalar data over shell in 2D model
    '''
    if (ndirs == 0):
        Vx.append (Vr)
        Vy.append (Vr)
        Vz.append (Vr)
    else:
        # Create ray directions
        (Rx, Ry, Rz) = rayVectors(dimension, ndirs)
        # Extend vectors
        for r in range (ndirs):
            Vx.append (Vr*Rx[r])
            Vy.append (Vr*Ry[r])
            Vz.append (Vr*Rz[r])


def mapToXD (model1D, dimension, nrays, cellsInShell):
    """
    Maps a 1D model to the spherically symmetric XD equivalent
    """
    # Create model and setup objects
    model = Model ()
    setup = Setup (dimension = dimension)
    # Get ncameras
    ncameras = len(model.geometry.cameras.camera2cell_nr)
    # Copy cameras from 1D model
    model.geometry.cameras = model1D.geometry.cameras
    # Add shells
    for s in range (model1D.parameters.ncells()+ncameras):
        if not s in model.geometry.cameras.camera2cell_nr:
            ndirs = len (cellsInShell[s])
            sphericalXDvector(model1D.geometry.cells.x[s],  dimension,     ndirs, model.geometry.cells.x,  model.geometry.cells.y,  model.geometry.cells.z)
            sphericalXDvector(model1D.geometry.cells.vx[s], dimension,     ndirs, model.geometry.cells.vx, model.geometry.cells.vy, model.geometry.cells.vz)
            sphericalXDscalar(model1D.chemistry.species.abundance[s],      ndirs, model.chemistry.species.abundance)
            sphericalXDscalar(model1D.thermodynamics.temperature.gas[s],   ndirs, model.thermodynamics.temperature.gas)
            sphericalXDscalar(model1D.thermodynamics.turbulence.vturb2[s], ndirs, model.thermodynamics.turbulence.vturb2)
    # Extract number of cells
    ncells = 0
    for shell in cellsInShell:
        ncells += len (shell)
    # Add Camera points
    for (cn,cp) in enumerate(model.geometry.cameras.camera2cell_nr):
        model.geometry.cells. x.append(model1D.geometry.cells. x[cp])
        model.geometry.cells. y.append(model1D.geometry.cells. y[cp])
        model.geometry.cells. z.append(model1D.geometry.cells. z[cp])
        model.geometry.cells.vx.append(model1D.geometry.cells.vx[cp])
        model.geometry.cells.vy.append(model1D.geometry.cells.vy[cp])
        model.geometry.cells.vz.append(model1D.geometry.cells.vz[cp])
        model.geometry.cameras.camera2cell_nr[cn] = ncells+cn
    # Extract boundary
    model.geometry.boundary.boundary2cell_nr = Long1 (cellsInShell[-1])
    # Add rays
    model.geometry.rays = setup.rays (nrays=nrays, cells=model.geometry.cells)
    # Set ncells and ncameras
    model.parameters.set_ncameras (ncameras)
    model.parameters.set_ncells   (ncells)
    # Extract neighbors
    model.geometry.cells = setup.neighborLists (cameras=model.geometry.cameras, cells=model.geometry.cells)
    # Add linedata
    model.lines = model1D.lines
    # Add chemical species
    model.chemistry.species.sym = model1D.chemistry.species.sym
    # Done
    return model
