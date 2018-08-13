import numpy as np
import healpy

from scipy.spatial        import Delaunay

def nRays(nsides):
    '''
    Number of rays corresponding to HEALPix's nsides
    '''
    return 12*nsides**2


def spherical3Dscalar(fr, nsides, f):
    nrays = nRays(nsides)
    f += [fr for _ in range(nrays)]


def spherical3Dvector(Vr, nsides, Vx, Vy, Vz):
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
