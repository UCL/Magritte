// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <omp.h>
#include <limits>

#include "declarations.hpp"


///  Constructor for CELLS: Allocates memory for cell data
///    @param number_of_cells: number of cells in grid
//////////////////////////////////////////////////////////

template <int Dimension, long Nrays>
CELLS <Dimension, Nrays> ::
CELLS (long number_of_cells)
{

  ncells = number_of_cells;

  x = new double[ncells];
  y = new double[ncells];
  z = new double[ncells];

  vx = new double[ncells];
  vy = new double[ncells];
  vz = new double[ncells];

  neighbor    = new long[ncells*Nrays];
  n_neighbors = new long[ncells];

  id      = new long[ncells];
  removed = new bool[ncells];

  boundary = new bool[ncells];
  mirror   = new bool[ncells];

}   // END OF CONSTRUCTOR



/// Destructor for CELLS: frees allocated memory
////////////////////////////////////////////////

template <int Dimension, long Nrays>
CELLS <Dimension, Nrays> ::
~CELLS ()
{

  delete [] x;
  delete [] y;
  delete [] z;

  delete [] vx;
  delete [] vy;
  delete [] vz;

  delete [] neighbor;
  delete [] n_neighbors;

  delete [] id;
  delete [] removed;

  delete [] boundary;
  delete [] mirror;

}   // END OF DESTRUCTOR



/// initialize: initialize cells with zeros or falses
/////////////////////////////////////////////////////

template <int Dimension, long Nrays>
int CELLS <Dimension, Nrays> ::
    initialize ()
{

# pragma omp parallel   \
  default (none)
  {

  int num_threads = omp_get_num_threads();
  int thread_num  = omp_get_thread_num();

  long start = (thread_num*ncells)/num_threads;
  long stop  = ((thread_num+1)*ncells)/num_threads;   // Note brackets


  for (long p = start; p < stop; p++)
  {
    x[p] = 0.0;
    y[p] = 0.0;
    z[p] = 0.0;

    n_neighbors[p] = 0;

    for (long r = 0; r < Nrays; r++)
    {
      neighbor[RINDEX(p,r)] = 0;
    }

    vx[p] = 0.0;
    vy[p] = 0.0;
    vz[p] = 0.0;

    id[p] = p;

    removed[p]  = false;
    boundary[p] = false;
    mirror[p]   = false;
  }
  } // end of OpenMP parallel region


  return (0);

}



///  next: find number of next cell on ray and its distance along ray
///    @param[in] origin: number of cell from which the ray originates
///    @param[in] r: number of the ray along which we are looking
///    @param[in] current: number of the cell put last on the ray
///    @param[in/out] *Z: pointer to the current distance along the ray
///    @param[out] *dZ: pointer to the distance increment to the next ray
///    @return number of the next cell on the ray after the current cell
/////////////////////////////////////////////////////////////////////////

template <int Dimension, long Nrays>
long CELLS <Dimension, Nrays> ::
     next (long origin, long r, long current, double *Z, double *dZ)
{

  // Pick neighbor on "right side" closest to ray

  double D_min = std::numeric_limits <double> :: max();   // Initialize to "infinity"

  long next = ncells;   // return ncells when there is no next cell


  for (long n = 0; n < n_neighbors[current]; n++)
  {
    long nb = neighbor[RINDEX(current,n)];

    double rvec[3];

    rvec[0] = x[nb] - x[origin];
    rvec[1] = y[nb] - y[origin];
    rvec[2] = z[nb] - z[origin];

    double Z_new =   rvec[0]*rays.x[r]
                   + rvec[1]*rays.y[r]
                   + rvec[2]*rays.z[r];

    if (*Z < Z_new)
    {
      double rvec2 = rvec[0]*rvec[0] + rvec[1]*rvec[1] + rvec[2]*rvec[2];

      double D = rvec2 - Z_new*Z_new;

      if (D < D_min)
      {
        D_min = D;
        next  = nb;
        *dZ   = Z_new - *Z;   // such that dZ > 0.0
      }
    }

  } // end of n loop over neighbors


  // Update distance along ray

  *Z = *Z + *dZ;


  return next;

}



///  relative_velocity: get relative velocity of current w.r.t. origin along ray
///    @param[in] origin: number of cell from which the ray originates
///    @param[in] r: number of the ray along which we are looking
///    @param[in] current: number of the cell for which we want the velocity
///    @return relative velocity of cell current w.r.t. cell origin
////////////////////////////////////////////////////////////////////////////////

template <int Dimension, long Nrays>
double CELLS <Dimension, Nrays> ::
       relative_velocity (long origin, long r, long current)
{

  return   (vx[current] - vx[origin]) * rays.x[r]
         + (vy[current] - vy[origin]) * rays.y[r]
         + (vz[current] - vz[origin]) * rays.z[r];

}
