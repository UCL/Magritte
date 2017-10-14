/* Frederik De Ceuster - University College London & KU Leuven                                   */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/* calc_LTE_populations: Calculates the LTE level populations                                    */
/*                                                                                               */
/* (NEW)                                                                                         */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/



#include <stdio.h>
#include <math.h>

#include "declarations.hpp"
#include "calc_LTE_populations.hpp"


/* calc_LTE_populations: Calculates the LTE level populations                                    */
/*-----------------------------------------------------------------------------------------------*/

int calc_LTE_populations( GRIDPOINT *gridpoint, double *energy, double *weight,
                          double *temperature_gas, double *pop )
{


  /* For each line producing species at each grid point */

  for (int lspec=0; lspec<NLSPEC; lspec++){

    for (long n=0; n<NGRID; n++){


      /* Calculate the partition function */

      double partition_function = 0.0;

      double total_population = 0.0;


      for (int i=0; i<nlev[lspec]; i++){

        int l_i = LSPECLEV(lspec,i);

        partition_function = partition_function
                             + weight[l_i] * exp( -energy[l_i]/(KB*temperature_gas[n]) );

      } /* end of i loop over levels */


      /* Calculate LTE level populations */

      for (int i=0; i<nlev[lspec]; i++){

        long p_i = LSPECGRIDLEV(lspec,n,i);
        int  l_i = LSPECLEV(lspec,i);

        pop[p_i] = gridpoint[n].density * weight[l_i] * exp( -energy[l_i]/(KB*temperature_gas[n]) )
                   / partition_function;

        total_population = total_population + pop[p_i];


        /* Avoid too small numbers */

        if (pop[p_i] < POP_LOWER_LIMIT){

          pop[p_i] = 0.0;
        }

      } /* end of i loop over levels */

      /* Check if total population adds up to the density */

      if ( (total_population-gridpoint[n].density)/total_population > 1.0E-3 ){

        printf("\nERROR : total of level populations differs from density !\n\n");
      }

    } /* end of n loop over grid points */

  } /* end of lspec loop over line producin species */


  return(0);
}

/*-----------------------------------------------------------------------------------------------*/
