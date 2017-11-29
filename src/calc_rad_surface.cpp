/* Frederik De Ceuster - University College London & KU Leuven                                   */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/* calc_rad_surface: Calculates the UV radiation surface for each ray at each grid point   */
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

#include <string>

#include "../parameters.hpp"
#include "Magritte_config.hpp"
#include "declarations.hpp"

#include "calc_rad_surface.hpp"



/* calc_rad_surface: calculates the UV radiation surface for each ray at each grid point         */
/*-----------------------------------------------------------------------------------------------*/

int calc_rad_surface(double *G_external, double *rad_surface)
{


  /* For all grid points */

# pragma omp parallel                                                                             \
  shared( G_external, rad_surface, unit_healpixvector )                                           \
  default( none )
  {

  int num_threads = omp_get_num_threads();
  int thread_num  = omp_get_thread_num();

  long start = (thread_num*NGRID)/num_threads;
  long stop  = ((thread_num+1)*NGRID)/num_threads;       /* Note the brackets are important here */


  for (long n=start; n<stop; n++){


    /* In case of a UNIdirectional radiation field */

    if (FIELD_FORM == "UNI"){


      /* Find the ray corresponding to the direction of G_external */

      double max_product = 0.0;

      double product;

      long r_max;


      for (long r=0; r<NRAYS; r++){

        rad_surface[RINDEX(n,r)] = 0.0;

        product = - ( G_external[0]*unit_healpixvector[VINDEX(r,0)]
                      + G_external[1]*unit_healpixvector[VINDEX(r,1)]
                      + G_external[2]*unit_healpixvector[VINDEX(r,2)] );

        if (product > max_product){

          max_product = product;

          r_max = r;
        }
      }

      rad_surface[RINDEX(n,r_max)] = max_product;

    } /* end if UNIform radiation field */


    /* In case of an ISOtropic radiation field */

    if (FIELD_FORM == "ISO"){


      for (long r=0; r<NRAYS; r++){

        rad_surface[RINDEX(n,r)] = G_external[0] / (double) NRAYS;
      }

    } /* end if ISOtropic radiation field */


  } /* end of n loop over grid points */
  } /* end of OpenMP parallel region */


  return(0);

}

/*-----------------------------------------------------------------------------------------------*/
