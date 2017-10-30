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

#include "../parameters.hpp"
#include "Magritte_config.hpp"
#include "declarations.hpp"

#include "calc_AV.hpp"



/* calc_AV: calculates the visual extinction along a ray ray at a grid point                     */
/*-----------------------------------------------------------------------------------------------*/

int calc_AV( double *column_H, double *AV )
{


  double A_V0 = 6.289E-22*metallicity;                  /* AV_fac in 3D-PDR code (A_V0 in paper) */



  /* For all grid points n and rays r */

  for (long n=0; n<NGRID; n++){

    for (long r=0; r<NRAYS; r++){

      AV[RINDEX(n,r)] = A_V0 * column_H[RINDEX(n,r)];
    }
  }


  return(0);

}

/*-----------------------------------------------------------------------------------------------*/
