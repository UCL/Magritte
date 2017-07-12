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



/* column_density_calculator: calculates the column density for each ray at each grid point      */
/*-----------------------------------------------------------------------------------------------*/

void column_density_calculator( EVALPOINT *evalpoint, double *column_density )
{

  long n1, n2;                                                               /* grid point index */

  long r;                                                                           /* ray index */

  long e;                                                              /* evaluation point index */

  long spec;                                                                    /* species index */



  /* Initialize the UV_field */

  for (n1=0; n1<ngrid; n1++){

    UV_field[n1] = 0.0;
  }



  /* For all grid points */

  for (n2=0; n2<ngrid; n2++){


    /* For all rays */

    for (r=0; r<NRAYS; r++){


      /* For all species */

      for (spec=0; spec<nspec; spec++){

          evnr = GP_NR_OF_EVALP(n2,r,1);

          column_density[GRADSPECRAY(n2,spec,r)] = column_density[GRADSPECRAY(n2,spec,r)]
                                                   + evalpoint[GINDEX(n2,evnr)].dZ
                                                     *( density[n2]*abundance[n2]
                                                        + density[evnr]*abundance[evnr] )/2.0;


        /* For every evaluation point along the ray (line of sight) */

        for (e=1; e<raytot[RINDEX(n2,r)]; e++){

          evnr = GP_NR_OF_EVALP(n2,r,e);

          column_density[GRADSPECRAY(n2,spec,r)] = column_density[GRADSPECRAY(n2,spec,r)]
                                                   + evalpoint[GINDEX(n2,evnr)].dZ
                                                     *( density[evnr-1]*abundance[evnr-1]
                                                        + density[evnr]*abundance[evnr] )/2.0;



        } /* end of e loop over evaluation points */

      } /* end of spec loop over species */

    } /* end of r loop over rays */

  } /* end of n2 loop over grid points */

}

/*-----------------------------------------------------------------------------------------------*/
