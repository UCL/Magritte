import numpy as np
import spheres

from healpy        import pixelfunc
from scipy.spatial import Delaunay


class model ():
    """
    Model data for Magritte
    """


    def __init__ (self, dim):
        """
        Constructor for model
        """
        self.dimension = dim
        # Number of cells
        self.ncells = 0
        # Number of rays
        self.nrays = 0
        # Cell center coordinates
        self.x = []
        self.y = []
        self.z = []
        # Cell velocity vectors
        self.vx = []
        self.vy = []
        self.vz = []
        # Ray vectors
        self.rx = []
        self.ry = []
        self.rz = []
        # Neighbor lists
        self.nNeighbors = []
        self.neighbors  = []
        # H2 Density
        self.density = []
        # Abundance
        self.abundance = []
        # Temperature [K]
        self.temperature = []
        # vturbulence [m/s]
        self.vturbulence = []
        # Boundary
        self.boundary = []


    def defineRays (self, nrays):
        """
        Define the directions of the rays
        """
        # Set number of rays
        self.nrays = nrays
        # Define directions of rays
        (self.rx, self.ry, self.rz) = spheres.rayVectors(self.dimension, nrays)


    def getNeighborLists (self):
        """
        Extract neighbor lists from cell centers assuming Voronoi tesselation
        """
        if   (self.dimension == 0):
            self.nNeighbors = [0]
        elif (self.dimension == 1):
            # For the middle points
            self.neighbors  = [[i-1, i+1] for i in range(1,self.ncells-1)]
            self.nNeighbors = [2          for i in range(1,self.ncells-1)]
            # For the end points
            self.neighbors  = [[1]] +  self.neighbors + [[self.ncells-2]]
            self.nNeighbors =  [1]  + self.nNeighbors +  [1]
        elif (self.dimension == 2):
            points  = [[self.x[i], self.y[i]] for i in range(self.ncells)]
            # Make a Delaulay triangulation
            delaunay = Delaunay(points)
            # Extract Delaunay vertices (= Voronoi neighbors)
            (indptr,indices) = delaunay.vertex_neighbor_vertices
            self.neighbors   = [indices[indptr[k]:indptr[k+1]] for k in range(self.ncells)]
            # Extract the number of neighbors for each point
            self.nNeighbors  = [len(neighborList) for neighborList in self.neighbors]
        elif (self.dimension == 3):
            points  = [[self.x[i], self.y[i], self.z[i]] for i in range(self.ncells)]
            # Make a Delaulay triangulation
            delaunay = Delaunay(points)
            # Extract Delaunay vertices (= Voronoi neighbors)
            (indptr,indices) = delaunay.vertex_neighbor_vertices
            self.neighbors   = [indices[indptr[k]:indptr[k+1]] for k in range(self.ncells)]
            # Extract the number of neighbors for each point
            self.nNeighbors  = [len(neighborList) for neighborList in self.neighbors]
        else:
            print ('ERROR: dimension not set!')


    def writeInput (self, folder):
        """
        Write Magritte input files to folder
        """
        nspec = 5
        zeros = np.zeros(self.ncells)
        ones  = np.ones(self.ncells)
        # Group all data
        txtgrid = np.stack((self.x, self.y, self.z, self.vx, self.vy, self.vz),         axis=1)
        txtrays = np.stack((self.rx, self.ry, self.rz),                                 axis=1)
        txtabun = np.stack((zeros, self.abundance, self.density, self.abundance, ones), axis=1)
        # Print to txt files
        np.savetxt(folder + '/cells.txt',       txtgrid,           fmt=6*'%lE\t')
        np.savetxt(folder + '/rays.txt',        txtrays,           fmt=3*'%lE\t')
        np.savetxt(folder + '/n_neighbors.txt', self.nNeighbors,   fmt='%ld')
        np.savetxt(folder + '/abundance.txt',   txtabun,           fmt=nspec*'%lE\t')
        np.savetxt(folder + '/temperature.txt', self.temperature,  fmt='%lE')
        np.savetxt(folder + '/vturbulence.txt', self.vturbulence,  fmt='%lE')
        np.savetxt(folder + '/boundary.txt',    self.boundary,     fmt='%ld')
        # Different format for neighbors which was variable line lengths
        with open(folder + '/neighbors.txt', 'w') as file:
            for p in range(self.ncells):
                line = ''
                if (self.nNeighbors[p] > 0):
                    for neighbor in self.neighbors[p]:
                        line += '{}\t'.format(neighbor)
                line += '\n'
                file.write(line)
                
                
    def readInput (self, folder):
        """
        Read Magritte input files from folder
        """
        (self.x, self.y, self.z, self.vx, self.vy, self.vz)  = np.loadtxt(folder + '/cells.txt',     unpack=True)
        (self.rx, self.ry, self.rz)                          = np.loadtxt(folder + '/rays.txt',      unpack=True)
        self.nNeighbors                                      = np.loadtxt(folder + '/n_neighbors.txt'           )
        (z, self.abundance, self.density, self.abundance, o) = np.loadtxt(folder + '/abundance.txt', unpack=True)
        self.temperature                                     = np.loadtxt(folder + '/temperature.txt'           )
        self.vturbulence                                     = np.loadtxt(folder + '/vturbulence.txt'           )
        self.boundary                                        = np.loadtxt(folder + '/boundary.txt'              )
        # Different format for neighbors which was variable line lengths
        with open(folder + '/neighbors.txt', 'w') as file:
            for line in file:
                sline = line.split()
                self.neighbors = [int(n) for n in sline]


    def imageRay (self, rayNumber):
        # Define ray
        Rx = self.rx[rayNumber]
        Ry = self.ry[rayNumber]
        Rz = self.rz[rayNumber]
        # Define help quantity
        sRx2pRy2 = np.sqrt(Rx**2 + Ry**2)
        if (sRx2pRy2 != 0.0):
            # Define unit vector along horizontal image axis 
            Ix =  Ry / sRx2pRy2
            Iy = -Rx / sRx2pRy2
            Iz =  0.0  
            # Define unit vector alonng vertical image axis 
            Jx = Rx * Rz / sRx2pRy2
            Jy = Ry * Rz / sRx2pRy2
            Jz = -sRx2pRy2
            # Define image coordinates 
            self.imageX = [self.x[p]*Ix + self.y[p]*Iy                for p in range(self.ncells)] 
            self.imageY = [self.x[p]*Jx + self.y[p]*Jy + self.z[p]*Jz for p in range(self.ncells)] 
        else:
            # Define image coordinates 
            self.imageX = self.x[p] 
            self.imageY = self.y[p] 


#

def mapToXD (model1D, dimension, nraysList):
    """
    Maps a 1D model to the spherically symmetric XD equivalent
    """
    # Create a 3D model object
    modelXD = model (dim=dimension)
    # Store cells number of cells in each cell
    cellsInShell = [[] for _ in range(model1D.ncells)]
    # Add shells
    index  = 0
    for s in range(model1D.ncells):
        nrays = nraysList[s]
        spheres.sphericalXDvector(model1D.x[s],  dimension, nrays, modelXD.x,  modelXD.y,  modelXD.z)
        spheres.sphericalXDvector(model1D.vx[s], dimension, nrays, modelXD.vx, modelXD.vy, modelXD.vz)
        spheres.sphericalXDscalar(model1D.density[s],       nrays, modelXD.density)
        spheres.sphericalXDscalar(model1D.abundance[s],     nrays, modelXD.abundance)
        spheres.sphericalXDscalar(model1D.temperature[s],   nrays, modelXD.temperature)
        spheres.sphericalXDscalar(model1D.vturbulence[s],   nrays, modelXD.vturbulence)
        if (nrays == 0):
            cellsInShell[s].append(index)
            index += 1
        else:
            for _ in range(nrays):
                cellsInShell[s].append(index)
                index += 1
    # Extract boundary 
    modelXD.boundary = cellsInShell[-1]
    # Extract number of cells
    modelXD.ncells   = index
    # Extract neighbors
    modelXD.getNeighborLists()
    # Done
    return (modelXD, cellsInShell)


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

