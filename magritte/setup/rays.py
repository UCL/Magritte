# Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
#
# Developed by: Frederik De Ceuster - University College London & KU Leuven
# _________________________________________________________________________


import numpy as np

from scipy.spatial.transform import Rotation
from healpy                  import pixelfunc
from numba                   import jit
from multiprocessing         import Pool, cpu_count

from magritte.core import Rays


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
        raise ValueError ('No HEALPix compatible nrays (nrays = 12*nsides**2).')
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
            delta = np.random.uniform(0., 2.*np.pi)
        Rx = [np.cos((2.*np.pi*r)/nrays+delta) for r in range(nrays)]
        Ry = [np.sin((2.*np.pi*r)/nrays+delta) for r in range(nrays)]
        Rz = [0.                               for _ in range(nrays)]
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
    if (sRx2pRy2 > 0.):
        # Define unit vector along horizontal image axis
        Ix =  Ry / sRx2pRy2
        Iy = -Rx / sRx2pRy2
        Iz =  0.
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




@jit(nopython=True)
def all_unique_enough(rs):
    # For all pairs of points
    for i in range(rs.shape[0]):
        for j in range(i+1, rs.shape[0]):
            # Check if they are too close
            if (np.dot(rs[i],rs[j]) > cosin):
                nn   = rs[j] - rs[i]
                norm = np.linalg.norm(nn)
                if (norm > 0.0):
                    nn = nn * (angle / norm)
                else:
                    # nn is zero, pick other direction
                    if (rs[j][0] > cosin):
                        nn = np.array([0.0, angle, 0.0])
                    else:
                        nn = np.array([angle, 0.0, 0.0])
                # Move and normalize the problematic point
                rs[j] = rs[j] + nn
                rs[j] = rs[j] / np.linalg.norm(rs[j])
                antipod = rs.shape[0]-j-1
                rs[antipod] = rs[antipod] - nn
                rs[antipod] = rs[antipod] / np.linalg.norm(rs[antipod])
                return False
    return True


@jit(nopython=True)
def make_unique(rs):
    # Try 1000 times
    for i in range(1000):
        if all_unique_enough(rs):
            return
    # Give up
    return 'Failed'


@jit(nopython=True)
def solid_angle(a, b, c):
    '''
    Solid angle spanned by the triangle abc, assuming a, b and c lie on a unit sphere.
    '''
    return np.abs(2.0*np.arctan(np.dot(a, np.cross(b,c))
                                / (1.0 + np.dot(a,b) + np.dot(c,a) + np.dot(b,c))))


@jit(nopython=True)
def get_weight(vertices, centre, region):
    '''
    Get the weight for a region, given the list of vertices and its centre.
    '''
    weight = 0.0
    for i in range(region.shape[0]):
        weight += solid_angle (centre, vertices[region[i-1]], vertices[region[i]])
    return weight


sample_size = 50

def get_rays(uni, points, i):
    # Exclude the current point from the sample of points
    sample = np.delete(np.arange(ncells), i)
    # Randomly pick sample_size number of points
    sample = np.random.choice(sample, size=sample_size, replace=False)
    # Compute the ray vectors w.r.t. point
    rs = points[sample] - points[i]
    # Compute the norms
    norms = np.linalg.norm(rs, axis=-1)
    # Normalize the ray vectors
    rs = rs * np.outer(1.0/norms, np.ones(3))
    # Add the basis rays
    rs = np.concatenate((rs, uni, -np.flip(rs, axis=0)), axis=0)
    # Ensure that the rs are unique
    if (make_unique(rs) == 'Failed'):
        raise StopIteration('Not unique after 1000 iterations')
    # Generate Voronoi tesselation
    sv = SphericalVoronoi(rs).sort_vertices_of_regions()
    # Extract the points and vertices
    vertices = np.array(sv.vertices)
    weights  = np.zeros(rs.shape[0])
    for p in range(rs.shape[0]):
        weights[p] = get_weight(vertices, rs[p], np.array(sv.regions[p]))
    # Return rays and their weights
    return (points, weights)


# class Rays():
#     def __init__(self, uni, points):
#         self.uni    = uni
#         self.points = points
#     def get(self, i):
#         get_rays(self.uni, self.points, i)
#
# def mp_get_rays(rays):
#     with Pool(processes=cpu_count()) as pool:
#         result = pool.map(rays.get, range(rays.points.shape[0]), chunksize=1)


def setup_rays_spherical_symmetry(nextra, points=[]):

    nrays = 2*(1 + len(points) + nextra)

    # for i, ri in enumerate(points):

    Rx = [1.0]
    Ry = [0.0]
    Rz = [0.0]

    for j, rj in enumerate(points):
       Rx.append(ri / np.sqrt(ri**2 + rj**2))
       Ry.append(rj / np.sqrt(ri**2 + rj**2))
       Rz.append(0.0)

    angle_max   = np.arctan(Ry[-1] / Rx[-1])
    angle_extra = (0.5*np.pi - angle_max) / nextra

    for k in range(1, nextra):
        Rx.append(np.cos(angle_max + k*angle_extra))
        Ry.append(np.sin(angle_max + k*angle_extra))
        Rz.append(0.0)

    Rx.append(0.0)
    Ry.append(1.0)
    Rz.append(0.0)

    Wt = []
    for n in range(nrays//2):
        if   (n == 0):
            upper_x, upper_y = 0.5*(Rx[n]+Rx[n+1]), 0.5*(Ry[n]+Ry[n+1])
            lower_x, lower_y = Rx[ 0], Ry[ 0]
        elif (n == nrays/2-1):
            upper_x, upper_y = Rx[-1], Ry[-1]
            lower_x, lower_y = 0.5*(Rx[n]+Rx[n-1]), 0.5*(Ry[n]+Ry[n-1])
        else:
            upper_x, upper_y = 0.5*(Rx[n]+Rx[n+1]), 0.5*(Ry[n]+Ry[n+1])
            lower_x, lower_y = 0.5*(Rx[n]+Rx[n-1]), 0.5*(Ry[n]+Ry[n-1])

        Wt.append(  lower_x / np.sqrt(lower_x**2 + lower_y**2)
                  - upper_x / np.sqrt(upper_x**2 + upper_y**2) )

    inverse_double_total = 1.0 / (2.0 * sum(Wt))

    for n in range(nrays//2):
        Wt[n] = Wt[n] * inverse_double_total

    # Append the antipodal rays
    for n in range(nrays//2):
        Rx.append(-Rx[n])
        Ry.append(-Ry[n])
        Rz.append(-Rz[n])
        Wt.append( Wt[n])

    rays = Rays()
    rays.rays    = np.array((Rx,Ry,Rz)).transpose()
    rays.weights = np.array(Wt)

    return rays