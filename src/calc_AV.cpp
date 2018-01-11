// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <omp.h>

#include "../parameters.hpp"
#include "Magritte_config.hpp"
#include "declarations.hpp"

#include "calc_AV.hpp"



// calc_AV: calculates visual extinction along a ray ray at a grid point
// ---------------------------------------------------------------------

int calc_AV (long ncells, double *column_tot, double *AV)
{

  const double A_V0 = 6.289E-22*METALLICITY;   // AV_fac in 3D-PDR code (A_V0 in paper)


  // For all grid points n and rays r

# pragma omp parallel              \
  shared(ncells, column_tot, AV)   \
  default( none )
  {

  int num_threads = omp_get_num_threads();
  int thread_num  = omp_get_thread_num();

  long start = (thread_num*NCELLS)/num_threads;
  long stop  = ((thread_num+1)*NCELLS)/num_threads;   // Note brackets


  for (long n = start; n < stop; n++){

    for (long r = 0; r < NRAYS; r++){

      AV[RINDEX(n,r)] = A_V0 * column_tot[RINDEX(n,r)];
    }

  } // end of n loop over grid points
  } // end of OpenMP parallel region


  return(0);

}
