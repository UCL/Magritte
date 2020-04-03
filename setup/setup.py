# Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
#
# Developed by: Frederik De Ceuster - University College London & KU Leuven
# _________________________________________________________________________

# General import
import numpy as np
import time
import re

from scipy.spatial import Delaunay
from rays          import rayVectors

# Magritte specific imports
from magritte import Linedata, CollisionPartner, LineProducingSpecies
from magritte import Rays
from magritte import vCollisionPartner


# Physical constants
CC = 2.99792458E+8    # [m/s] speed of light
HH = 6.62607004E-34   # [J*s] Planck's constant
KB = 1.38064852E-23   # [J/K] Boltzmann's constant

from random import randint
from math   import isclose

RT = 1.0E-5
#
# def get_rays (cells, nr, nrays):
#     ncells = len(cells.x)
#     (Rx, Ry, Rz) = rayVectors (dimension=3, nrays=int(80))
#     #while (len(Rx) < 300):
#     #    p = randint (0, ncells-1)
#     #    if (p != nr):
#     #        x = cells.x[p] - cells.x[nr]
#     #        y = cells.y[p] - cells.y[nr]
#     #        z = cells.z[p] - cells.z[nr]
#     #        length = np.sqrt(x**2 + y**2 + z**2)
#     #        x /= length
#     #        y /= length
#     #        z /= length
#     #        already_in_list = False
#     #        for r in range (len(Rx)):
#     #            if (isclose(x, Rx[r], abs_tol=RT) and isclose(y, Ry[r], abs_tol=RT) and isclose(z, Rz[r], abs_tol=RT)):
#     #                already_in_list = True
#     #        if not already_in_list:
#     #            # Add ray
#     #            Rx.append (+x)
#     #            Ry.append (+y)
#     #            Rz.append (+z)
#     #            # Add antipodal
#     #            Rx.append (-x)
#     #            Ry.append (-y)
#     #            Rz.append (-z)
#     return (Rx, Ry, Rz)
#

def get_neighbors(px, py, nr):
    x      =  px[nr]
    y      =  py[nr]
    x_orth = -py[nr]
    y_orth = +px[nr]
    n1 = nr
    s1 = 5.0
    n2 = nr
    s2 = 5.0
    for p in range(len(px)):
        rx = px[p]-x
        ry = py[p]-y
        s = rx**2 + ry**2
        n = x_orth*px[p] + y_orth*py[p]
        if (n > 0.0 and s < s1):
            n1 = p
            s1 = s
        if (n < 0.0 and s < s2):
            n2 = p
            s2 = s
    return (n1, n2)


def model_name ():
    """
    Get a date stamp to name the model
    """
    # Get a datestamp for the name
    dateStamp = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
    # Done
    return f'model_{dateStamp}'


class Setup ():

    '''
    Setup class for Magritte models.
    '''

    def __init__ (self, dimension):
        '''
        Constructor for setup.
        '''
        # Set (geometrical) dimension of the model
        self.dimension = dimension
        # Check validity of dimension
        if not self.dimension in [1, 3]:
            raise ValueError ('Dimension should be 1 or 3 (for now).')


#    def rays_new (self, nrays, cells):
#        """
#        Setup input for the Rays class.
#        """
#        # Check if nrays is a strictly positive integer
#        if not (nrays > 0):
#            raise ValueError('nrays should be strictly positive.')
#        if not isinstance (nrays, int):
#            raise ValueError('nrays should be an integer.')
#        # Set length
#        ncells = len (cells.x)
#        # Check lengths
#        assert(ncells == len(cells.y))
#        assert(ncells == len(cells.z))
#        
#        points = np.array([cells.x, cells.y, cells.z]).transpose()
#        for point in points:
#            rs = points - point
#            # Compute the norms
#            norms = np.linalg.norm(rs, axis=-1)
#            # Normalize the vectors
#            rs *= np.outer(1.0/norms, np.ones(3))
#        
#
#        
#        # Create rays object
#        rays = Rays()
#        # Create basis rays
#        (bx, by, bz) = rayVectors (dimension=self.dimension, nrays=nbrays)
#        # Define ray directions
#        for i in range(ncells):
#            rays.x.append(Double1([])))
#            rays.y.append(Double1([])))
#            rays.z.append(Double1([])))
#        
#        (Rx, Ry, Rz) = rayVectors (dimension=self.dimension, nrays=nrays)
#        # Assign ray vectors
#        rays.x = Double1(Rx)
#        rays.y = Double1(Ry)
#        rays.z = Double1(Rz)
#        # Done
#        return rays

    def rays (self, nrays, cells):
        """
        Setup input for the Rays class.
        """
        # Check lengths
        ncells = len(cells.position)
        # Check for consistency
        assert (ncells == len(cells.velocity))

        if (self.dimension == 2):
            raise ValueError('2D not supported')
            # Assign rays to each cell
            for p in range(ncells):
                (rx, ry, rz) = get_rays (cells, p, ncells)
                Rx.append (rx)
                Ry.append (ry)
                Rz.append (rz)
                # Get weights
                weights = []
                for r in range(len(rx)):
                    #(n1, n2) = get_neighbors (rx, ry, r)
                    #cos = rx[n1]*rx[n2] + ry[n1]*ry[n2]
                    #if (n1 == r):
                    #    print("n1 == r")
                    #    print("p= ", p, "    r = ", r)
                    #if (n2 == r):
                    #    print("n2 == r")
                    #    print("p= ", p, "    r = ", r)
                    #if (cos == 1.0):
                    #    print(p, r, n1, n2)
                    #    print(rx[n1], rx[n2], ry[n1], ry[n2])
                    #weights.append (0.5*np.arccos(cos)/(2.0*np.pi))
                    weights.append (1.0/80.0)
                wt.append (weights)
                #length = np.sqrt (cells.x[p]**2 + cells.y[p]**2 + cells.z[p]**2)
                ## Set up parameters
                #relevant_angle = np.arctan(7.5E+14 / length)
                #ntheta_small = 30
                #dtheta_small = relevant_angle / ntheta_small
                #ntheta_large = 25
                #dtheta_large = (np.pi - dtheta_small*ntheta_small) / ntheta_large
                ## Set first ray
                #Rx.append ([-cells.x[p]])
                #Ry.append ([-cells.y[p]])
                #Rz.append ([-cells.z[p]])
                #wt.append ([dtheta_small / (2.0*np.pi)])
                ## Assuming 2D...
                #for _ in range (ntheta_small):
                #    Rx[p].append (Rx[p][-1]*np.cos(dtheta_small) - Ry[p][-1]*np.sin(dtheta_small))
                #    Ry[p].append (Rx[p][-1]*np.sin(dtheta_small) + Ry[p][-1]*np.cos(dtheta_small))
                #    Rz[p].append (0.0)
                #    wt[p].append (dtheta_small / (2.0*np.pi))
                #wt[p][-1] = 0.5 * (dtheta_small + dtheta_large / (2.0 * np.pi))
                #for _ in range (ntheta_large):
                #    Rx[p].append (Rx[p][-1]*np.cos(dtheta_large) - Ry[p][-1]*np.sin(dtheta_large))
                #    Ry[p].append (Rx[p][-1]*np.sin(dtheta_large) + Ry[p][-1]*np.cos(dtheta_large))
                #    Rz[p].append (0.0)
                #    wt[p].append (dtheta_large / (2.0*np.pi))
                #for r in range (ntheta_small + ntheta_large+1):
                #    Rx[p].append (-Rx[p][r])
                #    Ry[p].append (-Ry[p][r])
                #    Rz[p].append (0.0)
                #    wt[p].append ( wt[p][r])
                ## Rescale (determinant of rotation matrix is not exactly 1)
                #for r in range (len(Rx[p])):
                #    length = np.sqrt (Rx[p][r]**2 + Ry[p][r]**2)
                #    Rx[p][r] = Rx[p][r] / length
                #    Ry[p][r] = Ry[p][r] / length
                #    Rz[p][r] = Rz[p][r] / length
                #    wt[p][r] = wt[p][r] / sum(wt[p])
        else:
            Rx = rayVectors (dimension=self.dimension, nrays=nrays)[0]
            Ry = rayVectors (dimension=self.dimension, nrays=nrays)[1]
            Rz = rayVectors (dimension=self.dimension, nrays=nrays)[2]
            wt = [1.0/nrays for r in range(nrays)]
        # Create rays object
        rays = Rays ()
        # Assign ray vectors
        # rays.x       = Double2([Double1(Rx[p]) for p in range(ncells)])
        # rays.y       = Double2([Double1(Ry[p]) for p in range(ncells)])
        # rays.z       = Double2([Double1(Rz[p]) for p in range(ncells)])
        # rays.weights = Double2([Double1(wt[p]) for p in range(ncells)])
        rays.rays    = np.array((Rx,Ry,Rz)).transpose()
        rays.weights = wt
        # Done
        return rays

    def neighborLists (self, cells):
        """
        Extract neighbor lists from cell centers assuming Voronoi tesselation
        """
        # Get length
        ncells = len(cells.position)
        # Check for consistency
        assert (ncells == len(cells.velocity))
        # Find neighbors
        if   (self.dimension == 1):
            # For the middle points
            nbs   = [[p-1, p+1] for p in range(1,ncells-1)]
            n_nbs = [ 2         for _ in range(1,ncells-1)]
            # For the first point
            nbs.insert   (0, [1])
            n_nbs.insert (0, 1)
            # For the last point
            nbs.append   ([ncells-2])
            n_nbs.append (1)

        elif (self.dimension == 2):
            raise ValueError ('Dimension = 2 is not supported.')
            #points  = [[cells.x[p], cells.y[p]] for p in range(ncells)]
            ## Make a Delaulay triangulation
            #delaunay = Delaunay (points)
            ## Extract Delaunay vertices (= Voronoi neighbors)
            #(indptr, indices) = delaunay.vertex_neighbor_vertices
            #cells.neighbors   = Long2 ([Long1 (indices[indptr[k]:indptr[k+1]]) for k in #range(ncells)])
            ## Extract the number of neighbors for each point
            #cells.n_neighbors = Long1 ([len (nList) for nList in cells.neighbors])
        elif (self.dimension == 3):
            #points  = [[cells.x[p], cells.y[p], cells.z[p]] for p in range(ncells)]
            points = cells.position
            # Make a Delaulay triangulation
            delaunay = Delaunay (points)
            # Extract Delaunay vertices (= Voronoi neighbors)
            (indptr, indices) = delaunay.vertex_neighbor_vertices
            nbs = [indices[indptr[k]:indptr[k+1]] for k in range(ncells)]
            # Extract the number of neighbors for each point
            n_nbs = [len (nb) for nb in nbs]
        # Change neighbors into a rectangular array (necessary for hdf5)
        nbs_rect = np.zeros((ncells, max(n_nbs)), dtype=int).tolist()
        for p in range(len(nbs)):
            for i,nb in enumerate(nbs[p]):
                nbs_rect[p][i] = nb
        cells.neighbors   = nbs_rect
        cells.n_neighbors = n_nbs
        # Done
        return cells


def getProperName(name):
    '''
    Return the standard name for the species
    '''
    if name in ['e']: return 'e-'
    if name in ['pH2', 'oH2', 'p-H2', 'o-H2']: return 'H2'
    # If none of the above special cases, it should be fine
    return name


def getSpeciesNumber (species, name):
    '''
    Returns number of species given by 'name'
    '''
    # Note that there are dummy species in Magritte at places 0 and NLSPEC
    if isinstance (name, list):
        return [getSpeciesNumber (species,elem) for elem in name]
    else:
        for i in range (len (species.sym)):
            if (species.sym[i] == getProperName (name)):
                return i
        return 0


def extractCollisionPartner (fileName, line, species, elem):
    '''
    Returns collision partner and whether it is ortho or para (for H2)
    '''
    with open (fileName) as dataFile:
        data = dataFile.readlines ()
    partner   = re.findall (elem.replace ('+','\+')+'\s*[\+\-]?\s*([\w\+\-]+)\s*', data[line])[0]
    excess    = re.findall ('[op]\-?', partner)
    if (len (excess) > 0):
        orthoPara = re.findall ('[op]', partner)[0]
        partner   = partner.replace (excess[0],'')
    else:
        orthoPara = 'n'
    return [getSpeciesNumber (species, partner), orthoPara]


class Reader ():
    def __init__ (self ,fileName):
        self.fileName = fileName

    def readColumn (self, start, nElem, columnNr, type):
        '''
        Returns a column of data as a list
        '''
        with open (self.fileName) as dataFile:
            lineNr = 0
            column = []
            for line in dataFile:
                if (lineNr >= start) and (lineNr < start+nElem):
                    if type == 'float':
                        column.append (float(line.split()[columnNr]))
                    if type == 'int':
                        column.append (int  (line.split()[columnNr]))
                    if type == 'str':
                        column.append (str  (line.split()[columnNr]))
                lineNr += 1
            # if type == 'float':
            #     column = Double1 (column)
            # if type == 'int':
            #     column = Long1   (column)
        return column

    def extractCollisionPartner (self, line, species, elem):
        '''
        Returns collision partner and whether it is ortho or para (for H2)
        '''
        with open (self.fileName) as dataFile:
            data = dataFile.readlines ()
        partner   = re.findall (elem.replace ('+','\+')+'\s*[\+\-]?\s*([\w\+\-]+)\s*', data[line])[0]
        excess    = re.findall ('[op]\-?', partner)
        if (len (excess) > 0):
            orthoPara = re.findall ('[op]', partner)[0]
            partner   = partner.replace (excess[0],'')
        else:
            orthoPara = 'n'
        return [getSpeciesNumber (species, partner), orthoPara]


def linedata_from_LAMDA_file (fileName, species, config={}):
    """
    Read line data in LAMDA format
    """
    '''
    Note: Do not use the Magritte objects ld etc. this will kill performance. Hence, the copies.
    '''


    # Create Lindata object
    ld = Linedata()
    # Create reader for data file
    rd = Reader(fileName)
    # Read radiative data
    sym          = rd.readColumn(start= 1,      nElem=1,    columnNr=0, type='str')[0]
    num          = getSpeciesNumber(species, sym)
    mass         = rd.readColumn(start= 3,      nElem=1,    columnNr=0, type='float')[0]
    inverse_mass = float (1.0 / mass)
    nlev         = rd.readColumn(start= 5,      nElem=1,    columnNr=0, type='int')[0]
    energy       = rd.readColumn(start= 7,      nElem=nlev, columnNr=1, type='float')
    weight       = rd.readColumn(start= 7,      nElem=nlev, columnNr=2, type='float')
    nrad         = rd.readColumn(start= 8+nlev, nElem=1,    columnNr=0, type='int')[0]
    irad         = rd.readColumn(start=10+nlev, nElem=nrad, columnNr=1, type='int')
    jrad         = rd.readColumn(start=10+nlev, nElem=nrad, columnNr=2, type='int')
    A            = rd.readColumn(start=10+nlev, nElem=nrad, columnNr=3, type='float')

    # Change index range from [1, nlev] to [0, nlev-1]
    for k in range(nrad):
        irad[k] += -1
        jrad[k] += -1

    # Convert to SI units
    for i in range(nlev):
        # Energy from [cm^-1] to [J]
        energy[i] *= 1.0E+2*HH*CC

    ld.sym          = sym
    ld.num          = num
    ld.inverse_mass = inverse_mass
    ld.nlev         = nlev
    ld.energy       = energy
    ld.weight       = weight

    # Start reading collisional data
    nlr = nlev + nrad

    ncolpar = rd.readColumn(start=11+nlr, nElem=1,  columnNr=0, type='int')[0]
    ind     = 13 + nlr

    ld.ncolpar = ncolpar

    # Create list of CollisionPartners and cast to vCollisionPartner
    ld.colpar = vCollisionPartner ([CollisionPartner() for _ in range(ncolpar)])

    # Loop over the collision partners
    for c in range(ncolpar):
        num_col_partner = rd.extractCollisionPartner(line=ind, species=species, elem=sym)[0]
        orth_or_para_H2 = rd.extractCollisionPartner(line=ind, species=species, elem=sym)[1]
        ncol            = rd.readColumn(start=ind+2, nElem=1,    columnNr=0,   type='int')[0]
        ntmp            = rd.readColumn(start=ind+4, nElem=1,    columnNr=0,   type='int')[0]
        icol            = rd.readColumn(start=ind+8, nElem=ncol, columnNr=1,   type='int')
        jcol            = rd.readColumn(start=ind+8, nElem=ncol, columnNr=2,   type='int')
        # Change index range from [1, nlev] to [0, nlev-1]
        for k in range(ncol):
            icol[k] += -1
            jcol[k] += -1
        tmp = []
        Cd  = []
        for t in range (ntmp):
            tmp.append (rd.readColumn(start=ind+6, nElem=1,    columnNr=t,   type='float')[0])
            Cd .append (rd.readColumn(start=ind+8, nElem=ncol, columnNr=3+t, type='float'))
        # Convert to SI units
        for t in range(ntmp):
            for k in range(ncol):
                # Cd from [cm^3] to [m^3]
                Cd[t][k] *= 1.0E-6

        Ce = [[0.0 for _ in range(ncol)] for _ in range(ntmp)]
        for t in range(ntmp):
            for k in range(ncol):
                i = icol[k]
                j = jcol[k]
                Ce[t][k] = Cd[t][k] * weight[i]/weight[j] * np.exp( -(energy[i]-energy[j]) / (KB*tmp[t]) )

        ld.colpar[c].num_col_partner = num_col_partner
        ld.colpar[c].orth_or_para_H2 = orth_or_para_H2
        ld.colpar[c].ncol            = ncol
        ld.colpar[c].ntmp            = ntmp
        ld.colpar[c].icol            = icol
        ld.colpar[c].jcol            = jcol
        ld.colpar[c].tmp             = tmp
        ld.colpar[c].Cd              = Cd
        ld.colpar[c].Ce              = Ce

        ind += 9 + ncol


    # Limit to the specified lines if required
    if ('considered transitions' in config) and (config['considered transitions'] is not None):
        if not isinstance(config['considered transitions'], list):
            config['considered transitions'] = [config['considered transitions']]
        if (len(config['considered transitions']) > 0):
            print('Not considering all radiative transitions on the data file but only the specified ones!')
            nrad = len (config['considered transitions'])
            irad = [irad[k] for k in config['considered transitions']]
            jrad = [jrad[k] for k in config['considered transitions']]
            A    = [   A[k] for k in config['considered transitions']]



    # Set derived quantities
    Bs        = [0.0 for _ in range(nrad)]
    Ba        = [0.0 for _ in range(nrad)]
    frequency = [0.0 for _ in range(nrad)]
    for k in range(nrad):
        i = irad[k]
        j = jrad[k]
        frequency[k] = (energy[i]-energy[j]) / HH
        Bs[k]        = A[k] * CC**2 / (2.0*HH*(frequency[k])**3)
        Ba[k]        = weight[i]/weight[j] * Bs[k]

    ld.nrad      = nrad
    ld.irad      = irad
    ld.jrad      = jrad
    ld.A         = A
    ld.Bs        = Bs
    ld.Ba        = Ba
    ld.frequency = frequency

    # Create LineProducingSpecies object
    lspec          = LineProducingSpecies()
    lspec.linedata = ld

    # Done
    return lspec


from os import mkdir

def make_file_structure (model, modelName):
    '''
    Make file structure for a text based model.
    '''
    mkdir(modelName)
    mkdir(f'{modelName}/Geometry')
    mkdir(f'{modelName}/Geometry/Cells')
    mkdir(f'{modelName}/Geometry/Rays')
    mkdir(f'{modelName}/Geometry/Boundary')
    mkdir(f'{modelName}/Thermodynamics')
    mkdir(f'{modelName}/Thermodynamics/Temperature')
    mkdir(f'{modelName}/Thermodynamics/Turbulence')
    mkdir(f'{modelName}/Chemistry')
    mkdir(f'{modelName}/Chemistry/Species')
    mkdir(f'{modelName}/Lines')
    for l in range(model.parameters.nlspecs()):
        mkdir(f'{modelName}/Lines/LineProducingSpecies_{l}')
        mkdir(f'{modelName}/Lines/LineProducingSpecies_{l}/Linedata')
        for c in range(model.lines.lineProducingSpecies[l].linedata.ncolpar):
            mkdir(f'{modelName}/Lines/LineProducingSpecies_{l}/Linedata/CollisionPartner_{c}')
        mkdir(f'{modelName}/Lines/LineProducingSpecies_{l}/Quadrature')
    mkdir(f'{modelName}/Radiation')
    mkdir(f'{modelName}/Radiation/Frequencies')
    mkdir(f'{modelName}/Simulation')
    mkdir(f'{modelName}/Simulation/Image')
