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

#include <string>

#include "../parameters.hpp"
#include "Magritte_config.hpp"
#include "declarations.hpp"

#include "calc_rad_surface.hpp"



/* calc_rad_surface: calculates the UV radiation surface for each ray at each grid point   */
/*-----------------------------------------------------------------------------------------------*/

int calc_rad_surface(double *G_external, double *unit_healpixvector, double *rad_surface)
{

  long n;                                                                    /* grid point index */
  long r;                                                                           /* ray index */


  /* For all grid points */

  for (n=0; n<NGRID; n++){


    /* In case of a UNIdirectional radiation field */

    if (FIELD_FORM == "UNI"){


      /* Find the ray corresponding to the direction of G_external */

      double max_product = 0.0;

      double product;

      long r_max;


      for (r=0; r<NRAYS; r++){

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


      for (r=0; r<NRAYS; r++){

        rad_surface[RINDEX(n,r)] = G_external[0] / (double) NRAYS;
      }

    } /* end if ISOtropic radiation field */


  } /* end of n loop over grid points */


  return(0);

}

/*-----------------------------------------------------------------------------------------------*/
