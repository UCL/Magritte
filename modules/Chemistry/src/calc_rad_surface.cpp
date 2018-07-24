// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <omp.h>

#include <string>


#include "declarations.hpp"
#include "calc_rad_surface.hpp"


// calc_rad_surface: calculates UV radiation surface for each ray at each grid point
// ---------------------------------------------------------------------------------

int calc_rad_surface (long ncells, CELLS *cells, RAYS rays, double *G_external)
{


  // For all grid points

# pragma omp parallel                                  \
  shared (ncells, cells, G_external, rays)   \
  default (none)
  {

  int num_threads = omp_get_num_threads();
  int thread_num  = omp_get_thread_num();

  long start = (thread_num*NCELLS)/num_threads;
  long stop  = ((thread_num+1)*NCELLS)/num_threads;   // Note brackets


  for (long p = start; p < stop; p++)
  {

    // In case of a UNIdirectional radiation field

    if (FIELD_FORM == "UNI")
    {

      // Find ray corresponding to direction of G_external

      double max_product = 0.0;

      long r_max;


      for (long r = 0; r < NRAYS; r++)
      {
        cells->rad_surface[RINDEX(p,r)] = 0.0;

        double product = - G_external[0]*rays.x[r]
                         - G_external[1]*rays.y[r]
                         - G_external[2]*rays.z[r];

        if (product > max_product)
        {
          max_product = product;

          r_max = r;
        }
      }

      cells->rad_surface[RINDEX(p,r_max)] = max_product;

    } // end if UNIdirectional radiation field


    // In case of an ISOtropic radiation field

    if (FIELD_FORM == "ISO")
    {

      for (long r = 0; r < NRAYS; r++)
      {
        cells->rad_surface[RINDEX(p,r)] = G_external[0] / (double) NRAYS;
      }
      
    } // end if ISOtropic radiation field


  } // end of n loop over grid points
  } // end of OpenMP parallel region


  return(0);

}
