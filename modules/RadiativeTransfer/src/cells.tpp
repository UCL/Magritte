// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <omp.h>
#include <vector>
#include <string>
#include <fstream>
#include <limits>
using namespace std;

#include "ompTools.hpp"


///  Constructor for CELLS: Allocates memory for cell data
///    @param number_of_cells: number of cells in grid
//////////////////////////////////////////////////////////

template <int Dimension, long Nrays>
CELLS <Dimension, Nrays> ::
CELLS (const long   num_of_cells,
       const string n_neighbors_file)
  : ncells (num_of_cells)
{

  x.resize (ncells);
  y.resize (ncells);
  z.resize (ncells);

  vx.resize (ncells);
  vy.resize (ncells);
  vz.resize (ncells);

  n_neighbors.resize (ncells);
    neighbors.resize (ncells);

       id.resize (ncells);
  removed.resize (ncells);

  boundary.resize (ncells);
    mirror.resize (ncells);

  boundary2cell_nr.resize (ncells);
  cell2boundary_nr.resize (ncells);


  // Read number of neighbors

  ifstream nNeighborsFile (n_neighbors_file);

  for (long p = 0; p < ncells; p++)
  {
    nNeighborsFile >> n_neighbors[p];
  }


# pragma omp parallel   \
  default (none)
  {

  for (long p = OMP_start(ncells); p < OMP_stop(ncells); p++)
  {
    neighbors[p].resize (n_neighbors[p]);

    x[p] = 0.0;
    y[p] = 0.0;
    z[p] = 0.0;

    vx[p] = 0.0;
    vy[p] = 0.0;
    vz[p] = 0.0;

    for (long n = 0; n < n_neighbors[p]; n++)
    {
      neighbors[p][n] = 0;
    }

    id[p] = p;

    removed[p]  = false;
    boundary[p] = false;
    mirror[p]   = false;

    cell2boundary_nr[p] = ncells;
    boundary2cell_nr[p] = ncells;

  }
  } // end of OpenMP parallel region

}   // END OF CONSTRUCTOR




///  read: read the cells, neighbors and boundary files
///////////////////////////////////////////////////////

template <int Dimension, long Nrays>
int CELLS <Dimension, Nrays> ::
    read (const string cells_file,
          const string neighbors_file,
          const string boundary_file  )
{

  // Read cell centers and velocities

  ifstream cellsFile (cells_file);

  for (long p = 0; p < ncells; p++)
  {
    cellsFile >> x[p] >> y[p] >> z[p] >> vx[p] >> vy[p] >> vz[p];
  }
  

  // Convert velocities in m/s to fractions for C

  for (long p = 0; p < ncells; p++)
  {
    vx[p] = vx[p] / CC;
    vy[p] = vy[p] / CC;
    vz[p] = vz[p] / CC;
  }


  // Read nearest neighbors lists

  ifstream neighborsFile (neighbors_file);

  for (long p = 0; p < ncells; p++)
  {
    for (long n = 0; n < n_neighbors[p]; n++)
    {
      neighborsFile >> neighbors[p][n];
    }
  }


  // Read boundary list

  ifstream boundaryFile (boundary_file);

  long index = 0;
  long cell_nr;

  while (boundaryFile >> cell_nr)
  {
    boundary2cell_nr[index]   = cell_nr;
    cell2boundary_nr[cell_nr] = index;

    boundary[cell_nr] = true;

    index++;
  }

  nboundary = index;


  return (0);

}




///  next: find number of next cell on ray and its distance along ray
///    @param[in] origin: number of cell from which the ray originates
///    @param[in] r: number of the ray along which we are looking
///    @param[in] current: number of the cell put last on the ray
///    @param[in/out] Z: reference to the current distance along the ray
///    @param[out] dZ: reference to the distance increment to the next ray
///    @return number of the next cell on the ray after the current cell
/////////////////////////////////////////////////////////////////////////

template <int Dimension, long Nrays>
inline long CELLS <Dimension, Nrays> ::
            next (const long    origin,
                  const long    r,
                  const long    current,
                        double &Z,
                        double &dZ      ) const
{

  // Pick neighbor on "right side" closest to ray

  double D_min = numeric_limits<double> :: max();   // Initialize to "infinity"

  long next = ncells;   // return ncells when there is no next cell


  for (long n = 0; n < n_neighbors[current]; n++)
  {
    long nb = neighbors[current][n];

    double rvec[3];

    rvec[0] = x[nb] - x[origin];
    rvec[1] = y[nb] - y[origin];
    rvec[2] = z[nb] - z[origin];

    double Z_new =   rvec[0]*rays.x[r]
                   + rvec[1]*rays.y[r]
                   + rvec[2]*rays.z[r];

    if (Z < Z_new)
    {
      double rvec2 = rvec[0]*rvec[0] + rvec[1]*rvec[1] + rvec[2]*rvec[2];

      double D = rvec2 - Z_new*Z_new;

      if (D < D_min)
      {
        D_min = D;
        next  = nb;
        dZ    = Z_new - Z;   // such that dZ > 0.0
      }
    }

  } // end of n loop over neighbors


  // Update distance along ray

  Z = Z + dZ;


  return next;

}


/// on_ray

template <int Dimension, long Nrays>
inline long CELLS <Dimension, Nrays> ::
            on_ray (const long    origin,
                    const long    ray,
                          long   *cellNrs,
                          double *dZs     ) const
{
  long    n = 0;     // number of cells on the ray
  double  Z = 0.0;   // distance from origin (o)
  double dZ = 0.0;   // last increment in Z

  long nxt = next (origin, ray, origin, Z, dZ);


  if (nxt != ncells)   // if we are not going out of grid
  {
    cellNrs[n] = nxt;
        dZs[n] = dZ;

    n++;

    while (!boundary[nxt])   // while we have not hit the boundary
    {
      nxt = next (origin, ray, nxt, Z, dZ);

      cellNrs[n] = nxt;
          dZs[n] = dZ;

      n++;
    }
  }


  return n;

}




///  relative_velocity: get relative velocity of current w.r.t. origin along ray
///    @param[in] origin: number of cell from which the ray originates
///    @param[in] r: number of the ray along which we are looking
///    @param[in] current: number of the cell for which we want the velocity
///    @return relative velocity of cell current w.r.t. cell origin
////////////////////////////////////////////////////////////////////////////////

template <int Dimension, long Nrays>
inline double CELLS <Dimension, Nrays> ::
    doppler_shift         (
        const long origin,
        const long ray,
        const long current) const
{

  return 1.0 - (  (vx[current] - vx[origin]) * rays.x[ray]
                + (vy[current] - vy[origin]) * rays.y[ray]
                + (vz[current] - vz[origin]) * rays.z[ray]);

}



template <int Dimension, long Nrays>
inline double CELLS <Dimension, Nrays> ::
    x_projected       (
        const long p,
        const long r  ) const
{
  return x[p]*rays.Ix[r] + y[p]*rays.Iy[r];
}

template <int Dimension, long Nrays>
inline double CELLS <Dimension, Nrays> ::
    y_projected       (
        const long p,
        const long r  ) const
{
  return x[p]*rays.Jx[r] + y[p]*rays.Jy[r] + z[p]*rays.Jz[r];
}
