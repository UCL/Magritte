/* Frederik De Ceuster - University College London                                               */
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

#include "declarations.hpp"
#include "calc_AV.hpp"



/* calc_AV: calculates the visual extinction along a ray ray at a grid point               */
/*-----------------------------------------------------------------------------------------------*/

void calc_AV( double *column_H2, double *AV )
{

  long n;                                                                    /* grid point index */

  long r;                                                                           /* ray index */

  double A_V0 = 6.289E-22*metallicity;                  /* AV_fac in 3D-PDR code (A_V0 in paper) */



  /* For all grid points n and rays r */

  for (n=0; n<NGRID; n++){

    for (r=0; r<NRAYS; r++){

      AV[RINDEX(n,r)] = A_V0 * column_H2[RINDEX(n,r)];
    }
  }


}

/*-----------------------------------------------------------------------------------------------*/
