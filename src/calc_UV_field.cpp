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

#include "declarations.hpp"
#include "calc_UV_field.hpp"



/* calc_UV_field: calculates the UV radiation field at each grid point                           */
/*-----------------------------------------------------------------------------------------------*/

void calc_UV_field( double *AV, double *rad_surface, double *UV_field )
{

  long n;                                                                    /* grid point index */
  long r;                                                                           /* ray index */

  double tau_UV = 3.02;   /* dimensionless factor converting visual extinction to UV attenuation */



  /* For all grid points */

  for (n=0; n<NGRID; n++){


    UV_field[n] = 0.0;


    /* For all rays */

    for (r=0; r<NRAYS; r++){

      if(raytot[RINDEX(n,r)] > 0){

        UV_field[n] = UV_field[n] + rad_surface[RINDEX(n,r)]*exp(-tau_UV*AV[RINDEX(n,r)]);
      }
    }


  } /* end of n loop over grid points */


}

/*-----------------------------------------------------------------------------------------------*/
