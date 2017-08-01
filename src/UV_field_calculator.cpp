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

#include "declarations.hpp"
#include "UV_field_calculator.hpp"



/* UV_field_calculator: calculates the UV radiation field at each grid point                     */
/*-----------------------------------------------------------------------------------------------*/

void UV_field_calculator(double *G_external, double *UV_field, double *rad_surface)
{

  long n;                                                                    /* grid point index */

  long r;                                                                           /* ray index */



  /* For all grid points */

  for (n=0; n<NGRID; n++){


    /* For all rays */

    for (r=0; r<NRAYS; r++){

      UV_field[RINDEX(n,r)] = 0.0;

      rad_surface[RINDEX(n,r)] = G_external[0] / (double) NRAYS;
    }
  }

}

/*-----------------------------------------------------------------------------------------------*/
