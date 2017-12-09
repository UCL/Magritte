/* Frederik De Ceuster - University College London & KU Leuven                                   */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/* culumn_density_calculator: Calculates the column density along each ray at each grid point    */
/*                                                                                               */
/* (based on 3DPDR in 3D-PDR)                                                                    */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/



#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <omp.h>

#include "../parameters.hpp"
#include "Magritte_config.hpp"
#include "declarations.hpp"

#include "calc_AV.hpp"



/* calc_AV: calculates the visual extinction along a ray ray at a grid point                     */
/*-----------------------------------------------------------------------------------------------*/

int calc_AV( double *column_tot, double *AV )
{


  const double A_V0 = 6.289E-22*METALLICITY;            /* AV_fac in 3D-PDR code (A_V0 in paper) */


  /* For all grid points n and rays r */

# pragma omp parallel                                                                             \
  shared( column_tot, AV)                                                                         \
  default( none )
  {

  int num_threads = omp_get_num_threads();
  int thread_num  = omp_get_thread_num();

  long start = (thread_num*NGRID)/num_threads;
  long stop  = ((thread_num+1)*NGRID)/num_threads;       /* Note the brackets are important here */


  for (long n=start; n<stop; n++){

    for (long r=0; r<NRAYS; r++){

      AV[RINDEX(n,r)] = A_V0 * column_tot[RINDEX(n,r)];
    }

  } /* end of n loop over grid points */
  } /* end of OpenMP parallel region */


  return EXIT_SUCCESS;

}

/*-----------------------------------------------------------------------------------------------*/
