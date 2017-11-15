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

#include "calc_column_density.hpp"
#include "ray_tracing.hpp"



#ifndef ON_THE_FLY



/* calc_column_density: calculates column density for each species, ray and grid point           */
/*-----------------------------------------------------------------------------------------------*/

int calc_column_density( GRIDPOINT *gridpoint, EVALPOINT *evalpoint,
                          double *column_density, int spec )
{


  /* For all grid points n and rays r */

  for (long n=0; n<NGRID; n++){

    for (long r=0; r<NRAYS; r++){

      column_density[RINDEX(n,r)] = column_density_at_point(gridpoint, evalpoint, n, spec, r);

    }
  }


  return(0);

}

/*-----------------------------------------------------------------------------------------------*/





/* column_density: calculates the column density for one species along one ray                   */
/*-----------------------------------------------------------------------------------------------*/

double column_density_at_point( GRIDPOINT *gridpoint, EVALPOINT *evalpoint,
                                long gridp, int spec, long ray )
{


  double column_density_res = 0.0;                                   /* resulting column density */


  if(raytot[RINDEX(gridp,ray)] > 0){

    long evnr = GP_NR_OF_EVALP(gridp,ray,0);

    column_density_res = evalpoint[GINDEX(gridp,evnr)].dZ * PC
                         *( gridpoint[gridp].density*species[spec].abn[gridp]
                            + gridpoint[evnr].density*species[spec].abn[evnr] ) / 2.0;


    /* Numerical integration along the ray (line of sight) */

    for (long e=1; e<raytot[RINDEX(gridp,ray)]; e++){

      long evnr  = GP_NR_OF_EVALP(gridp,ray,e);
      long evnrp = GP_NR_OF_EVALP(gridp,ray,e-1);

      column_density_res = column_density_res
                           + evalpoint[GINDEX(gridp,evnr)].dZ * PC
                             * ( gridpoint[evnrp].density*species[spec].abn[evnrp]
                                 + gridpoint[evnr].density*species[spec].abn[evnr] ) / 2.0;

    } /* end of e loop over evaluation points */

  }

  return column_density_res;

}

/*-----------------------------------------------------------------------------------------------*/





#else





/* calc_column_density: calculates column density for each species, ray and grid point           */
/*-----------------------------------------------------------------------------------------------*/

int calc_column_density( GRIDPOINT *gridpoint, double *column_density, int spec )
{


  /* For all grid points n and rays r */

  for (long n=0; n<NGRID; n++){

    EVALPOINT local_evalpoint[NGRID];

    get_evalpoints(gridpoint, local_evalpoint, n);


    for (long r=0; r<NRAYS; r++){

      column_density[RINDEX(n,r)] = column_density_at_point(gridpoint, local_evalpoint, n, spec, r);

    }
  }


  return(0);

}

/*-----------------------------------------------------------------------------------------------*/





/* column_density: calculates the column density for one species along one ray                   */
/*-----------------------------------------------------------------------------------------------*/

double column_density_at_point( GRIDPOINT *gridpoint, EVALPOINT *local_evalpoint,
                                long gridp, int spec, long ray )
{


  double column_density_res = 0.0;                                   /* resulting column density */


  if(local_raytot[ray] > 0){

    long evnr = LOCAL_GP_NR_OF_EVALP(ray,0);

    column_density_res = evalpoint[evnr].dZ * PC
                         *( gridpoint[gridp].density*species[spec].abn[gridp]
                            + gridpoint[evnr].density*species[spec].abn[evnr] ) / 2.0;


    /* Numerical integration along the ray (line of sight) */

    for (long e=1; e<local_raytot[ray]; e++){

      long evnr  = LOCAL_GP_NR_OF_EVALP(ray,e);
      long evnrp = LOCAL_GP_NR_OF_EVALP(ray,e-1);

      column_density_res = column_density_res
                           + evalpoint[evnr].dZ * PC
                             * ( gridpoint[evnrp].density*species[spec].abn[evnrp]
                                 + gridpoint[evnr].density*species[spec].abn[evnr] ) / 2.0;

    } /* end of e loop over evaluation points */

  }

  return column_density_res;

}

/*-----------------------------------------------------------------------------------------------*/



#endif
