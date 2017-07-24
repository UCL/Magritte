/* Frederik De Ceuster - University College London                                               */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/* culumn_density_calculator: Calculates the UV radiation field at each grid point                     */
/*                                                                                               */
/* (based on 3DPDR in 3D-PDR)                                                                    */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/



#include <stdio.h>
#include <math.h>
#include <stdlib.h>



/* column_density_calculator: calculates column density for each species, ray and grid point     */
/*-----------------------------------------------------------------------------------------------*/

void column_density_calculator( GRIDPOINT *gridpoint, EVALPOINT *evalpoint,
                                double *column_density, double *AV )
{

  long n;                                                                    /* grid point index */

  long r;                                                                           /* ray index */

  long spec;                                                                    /* species index */


  double column_density_( GRIDPOINT *gridpoint, EVALPOINT *evalpoint,
                          long gridp, int spec, long ray );

  int get_species_nr(string name);

  int H_nr = get_species_nr("H");                               /* species nr corresponding to H */ 

  double A_V0 = 6.289E-22*metallicity;                  /* AV_fac in 3D-PDR code (A_V0 in paper) */



  /* For all grid points */

  for (n=0; n<NGRID; n++){


    /* For all rays */

    for (r=0; r<NRAYS; r++){


      /* For all species */

      for (spec=0; spec<NSPEC; spec++){

        column_density[GRIDSPECRAY(n,spec,r)] = column_density_(gridpoint, evalpoint ,n, spec, r);

        if ( spec == H_nr ){

          AV[RINDEX(n,r)] = A_V0 * column_density[GRIDSPECRAY(n,spec,r)];
        }

      } /* end of spec loop over species */

    } /* end of r loop over rays */

  } /* end of n2 loop over grid points */

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
