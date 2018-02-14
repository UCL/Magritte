// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <omp.h>

#include "declarations.hpp"
#include "calc_UV_field.hpp"


// calc_UV_field: calculates UV radiation field at each cell
// ---------------------------------------------------------

int calc_UV_field (long ncells, double *AV, double *rad_surface, double *UV_field)
{

  const double tau_UV = 3.02;   // conversion factor from visual extinction to UV attenuation


  // For all grid points

# pragma omp parallel                                   \
  shared (ncells, AV, rad_surface, UV_field, antipod)   \
  default (none)
  {

  int num_threads = omp_get_num_threads();
  int thread_num  = omp_get_thread_num();

  long start = (thread_num*NCELLS)/num_threads;
  long stop  = ((thread_num+1)*NCELLS)/num_threads;   // Note brackets


  for (long n = start; n < stop; n++)
  {
    UV_field[n] = 0.0;


    // External UV radiation field
    // _ _ _ _ _ _ _ _ _ _ _ _ _ _


    // For all rays

    for (long r = 0; r < NRAYS; r++)
    {
      long nr  = RINDEX(n,r);
      long nar = RINDEX(n,antipod[r]);

      UV_field[n] = UV_field[n] + rad_surface[nr]*exp(-tau_UV*AV[nr]);
    }




    // Internal UV radiation field
    // _ _ _ _ _ _ _ _ _ _ _ _ _ _


  } // end of n loop over grid points
  } // end of OpenMP parallel region


  return(0);

}
