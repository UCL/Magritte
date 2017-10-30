/* Frederik De Ceuster - University College London & KU Leuven                                   */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/* calc_UV_field: Calculates the UV radiation field at each grid point                           */
/*                                                                                               */
/* (based on 3DPDR in 3D-PDR)                                                                    */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/



#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#include "../parameters.hpp"
#include "Magritte_config.hpp"
#include "declarations.hpp"

#include "calc_UV_field.hpp"



/* calc_UV_field: calculates the UV radiation field at each grid point                           */
/*-----------------------------------------------------------------------------------------------*/

int calc_UV_field( long *antipod, double *AV, double *rad_surface, double *UV_field )
{


  double tau_UV = 3.02;   /* dimensionless factor converting visual extinction to UV attenuation */


  /* For all grid points */

  for (long n=0; n<NGRID; n++){

    UV_field[n] = 0.0;


    /* For all rays */

    for (long r=0; r<NRAYS; r++){

      long nr  = RINDEX(n,r);
      long nar = RINDEX(n,antipod[r]);

      // printf("r and ar are : %ld and %ld\n", r, antipod[r]);

      if ( (raytot[nr] > 0) || (raytot[nar] > 0) ){

        UV_field[n] = UV_field[n] + rad_surface[nr]*exp(-tau_UV*AV[nr]);
      }
    }


  } /* end of n loop over grid points */


  return(0);

}

/*-----------------------------------------------------------------------------------------------*/
