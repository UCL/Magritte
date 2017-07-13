/* Frederik De Ceuster - University College London                                               */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/* UV_field_calculator: Calculates the UV radiation field at each grid point                     */
/*                                                                                               */
/* (based on 3DPDR in 3D-PDR)                                                                    */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/



#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#include "column_density_calculator.cpp"
  


/* UV_field_calculator: calculates the UV radiation field at each grid point                     */
/*-----------------------------------------------------------------------------------------------*/

void UV_field_calculator(double *UV_field, double *AV)
{

  long n;                                                                    /* grid point index */

  long r;                                                                           /* ray index */

  double A_V0 = 6.289E-22*metallicity;                  /* AV_fac in 3D-PDR code (A_V0 in paper) */


  /* Initialize the UV_field */

  for (n=0; n<ngrid; n++){

    UV_field[n] = 0.0;
  }


  /* For all grid points */

  for (n=0; n<ngrid; n++){


  

    /* For all rays */

    for (r=0; r<NRAYS; r++){

      
    }

  }
}

/*-----------------------------------------------------------------------------------------------*/
