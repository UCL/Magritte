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



/* UV_field_calculator: calculates the UV radiation field at each grid point                     */
/*-----------------------------------------------------------------------------------------------*/

void UV_field_calculator(double *UV_field)
{

  long n;                                                                    /* grid point index */

  long r;                                                                           /* ray index */


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
