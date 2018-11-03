import numpy as np
import healpy

from scipy.spatial import Delaunay

from model import model


# Helper functions


def nRays (nsides):
    '''
    Number of rays corresponding to HEALPix's nsides
    '''
    return 12*nsides**2


def spherical3Dscalar (fr, nsides, f):
    '''
    Copy 1D model scalar data over shell in 3D model 
    '''
    if (nsides == 0):
        f += [fr]
    else:
        nrays = nRays(nsides)
        f += [fr for _ in range(nrays)]


def spherical3Dvector(Vr, nsides, Vx, Vy, Vz):
    '''
    Copy 1D model scalar data over shell in 3D model 
    '''
    if (nsides == 0):
        Vx += [Vr]
        Vy += [Vr]
        Vz += [Vr]
    else:
        nrays = nRays(nsides)
        (Rx, Ry, Rz) = healpy.pixelfunc.pix2vec(nsides, range(nrays))
        Vx += [Vr*rx for rx in Rx]
        Vy += [Vr*ry for ry in Ry]
        Vz += [Vr*rz for rz in Rz]

def neighborLists_assumingVoronoiTesselation(x,y,z):
    npoints = len(x)
    points = [[x[i], y[i], z[i]] for i in range(npoints)]
    # Make a Delaulay triangulation
    delaunay = Delaunay(points)
    # Extract Delaunay vertices (= Voronoi neighbors)
    (indptr,indices) = delaunay.vertex_neighbor_vertices
    neighborLists = [indices[indptr[k]:indptr[k+1]] for k in range(npoints)]
    # Done
    return neighborLists

def relDiff(a,b):
    if (a+b == 0.0):
        return 0.0
    else:
        return abs((a-b)/(a+b))

def deSphere(cellsInShell, f):
    (nOut, ncells, n) = np.shape(f)
    nshells = len(cellsInShell)
    fR = np.zeros((nOut,nshells,n))
    for o in range(nOut):
        for s in range(nshells):
            for p in cellsInShell[s]:
                fR[o][s] += f[o][p]
            fR[o][s] = fR[o][s] / len(cellsInShell[s])
    return fR


def deSphereMax(cellsInShell, f):
    (nOut, ncells, n) = np.shape(f)
    nshells = len(cellsInShell)
    fR = np.zeros((nOut,nshells,n))
    for o in range(nOut):
        for s in range(nshells):
            for i in range(n):
                fR[o][s][i] = np.max(f[o][cellsInShell[s]][i])
    return fR

def sphereVar(cellsInShell, f):
    (nOut, ncells, n) = np.shape(f)
    nshells = len(cellsInShell)
    fVar = np.zeros((nOut,nshells,n))
    fR = deSphere(cellsInShell, f)
    for o in range(nOut):
        for s in range(nshells):
            for p in cellsInShell[s]:
                for i in range(n):
                    fVar[o][s][i] += relDiff(f[o][p][i],fR[o][s][i])
            fVar[o][s] = fVar[o][s] / len(cellsInShell[s])
    return fVar


def mapTo3D (model1D, nsidesList):
    """
    Maps a 1D model to the spherically symmetric 3D equivalent
    """
    # Create a 3D model object
    model3D = model (dim=3)
    # Store cells number of cells in each cell
    cellsInShell = [[] for _ in range(model1D.ncells)]
    # Add shells
    index  = 0
    for s in range(model1D.ncells):
        nrays = nRays(nsidesList[s])
        spherical3Dvector(model1D.x[s],           nsidesList[s], model3D.x,  model3D.y,  model3D.z)
        spherical3Dvector(model1D.vx[s],          nsidesList[s], model3D.vx, model3D.vy, model3D.vz)
        spherical3Dscalar(model1D.density[s],     nsidesList[s], model3D.density)
        spherical3Dscalar(model1D.abundance[s],   nsidesList[s], model3D.abundance)
        spherical3Dscalar(model1D.temperature[s], nsidesList[s], model3D.temperature)
        if (nrays == 0):
            cellsInShell[s].append(index)
            index += 1
        else:
            for _ in range(nrays):
                cellsInShell[s].append(index)
                index += 1
    # Extract boundary 
    model3D.boundary = cellsInShell[-1]
    # Extract number of cells
    model3D.ncells   = index
    # Extract neighbors
    model3D.getNeighborLists()
    # Done
    return (model3D, cellsInShell)


def mapTo1D (model3D, model1D, cellsInShell):
    """
    Maps a spherically symmetric 3D model to 1D equivalent
    """
    return  
    
#def mapTo3D_interpolate (model1D, nEdgeCells, dEdgeLength)
#    """
#    Maps a 1D model to the spherically symmetric 3D equivalent
#    """
#    # Create a 3D model object
#    model3D = setCubeGrid (nEdgeCells, dEdgeLength)
#    for n in range(model3D.ncells):
#        r = np.sqrt (model3D.x[n]**2 + model3D.y[n]**2 + model3D.z[n]**2)
#        if r
#
#
#    return model3D


def setCubeGrid (nEdgeCells, dEdgeLength):
    """
    Create a 3D cubic Cartesian grid
    """
    # Create a 3D model object
    model3D = model (dim=3)
    # Add geometry to grid
    for i in range(nEdgeCells):
        for j in range(nEdgeCells):
            for k in range(nEdgeCells):
                model3D.x = i * edgeLength;
                model3D.y = j * edgeLength;
                model3D.z = k * edgeLength;
    # Done
    return model3D




def makeShelly (model1D, model3D):
    for n in range(model3D.ncells):
        r = np.sqrt (model3D.x[n]**2 + model3D.y[n]**2 + model3D.z[n]**2)
        for m in range(-model1D.ncells):
            if (model.x[m] < r):
                ratio = model.x[m] / r
                model3D.x[n] *= ratio
                model3D.y[n] *= ratio
                model3D.z[n] *= ratio
                break

