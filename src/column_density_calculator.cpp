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
#include "column_density_calculator.hpp"



/* column_density_calculator: calculates column density for each species, ray and grid point     */
/*-----------------------------------------------------------------------------------------------*/

void column_density_calculator( GRIDPOINT *gridpoint, EVALPOINT *evalpoint,
                                double *column_density, int spec )
{

  long n;                                                                    /* grid point index */

  long r;                                                                           /* ray index */

  /* For all grid points n and rays r */

  for (n=0; n<NGRID; n++){

    for (r=0; r<NRAYS; r++){

      column_density[RINDEX(n,r)] = column_density_(gridpoint, evalpoint, n, spec, r);

    }
  }


}

/*-----------------------------------------------------------------------------------------------*/





/* column_density: calculates the column density for one species along one ray                   */
/*-----------------------------------------------------------------------------------------------*/

double column_density_( GRIDPOINT *gridpoint, EVALPOINT *evalpoint,
                        long gridp, int spec, long ray )
{

  double column_density_res = 0.0;                                   /* resulting column density */

  long e, evnr;                                                        /* evaluation point index */



  evnr = GP_NR_OF_EVALP(gridp,ray,0);

  column_density_res = evalpoint[GINDEX(gridp,evnr)].dZ
                       *( gridpoint[gridp].density*species[spec].abn[gridp]
                          + gridpoint[evnr].density*species[spec].abn[evnr] ) / 2.0;


  /* Numerical integration along the ray (line of sight) */

  for (e=1; e<raytot[RINDEX(gridp,ray)]; e++){

    evnr = GP_NR_OF_EVALP(gridp,ray,e);

    column_density_res = column_density_res
                         + evalpoint[GINDEX(gridp,evnr)].dZ
                           * ( gridpoint[evnr-1].density*species[spec].abn[evnr-1]
                               + gridpoint[evnr].density*species[spec].abn[evnr] ) / 2.0;

  } /* end of e loop over evaluation points */


  return column_density_res;

}

/*-----------------------------------------------------------------------------------------------*/
