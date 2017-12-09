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
#include <omp.h>

#include "../parameters.hpp"
#include "Magritte_config.hpp"
#include "declarations.hpp"

#include "calc_LTE_populations.hpp"


/* calc_LTE_populations: Calculates the LTE level populations                                    */
/*-----------------------------------------------------------------------------------------------*/

int calc_LTE_populations( GRIDPOINT *gridpoint, double *energy, double *weight,
                          double *temperature_gas, double *pop )
{


  /* For each line producing species at each grid point */

  for (int lspec=0; lspec<NLSPEC; lspec++){

#   pragma omp parallel                                                                           \
    shared( gridpoint, energy, weight, temperature_gas, pop, nlev, cum_nlev, species, lspec_nr,   \
            lspec )                                                                               \
    default( none )
    {

    int num_threads = omp_get_num_threads();
    int thread_num  = omp_get_thread_num();

    long start = (thread_num*NGRID)/num_threads;
    long stop  = ((thread_num+1)*NGRID)/num_threads;          /* Note the brackets are important */


    for (long n=start; n<stop; n++){


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

        pop[p_i] = gridpoint[n].density * species[lspec_nr[lspec]].abn[n] * weight[l_i]
                   * exp( -energy[l_i]/(KB*temperature_gas[n]) ) / partition_function;

        total_population = total_population + pop[p_i];


        /* Avoid too small numbers */

        // if (pop[p_i] < POP_LOWER_LIMIT){
        //
        //   pop[p_i] = 0.0;
        // }

      } /* end of i loop over levels */

      /* Check if total population adds up to the density */

      if ( (total_population-gridpoint[n].density*species[lspec_nr[lspec]].abn[n])/total_population > 1.0E-3 ){

        printf("\nERROR : total of level populations differs from density !\n\n");
      }

    } /* end of n loop over grid points */
    } /* end of OpenMP parallel region */

  } /* end of lspec loop over line producin species */


  return EXIT_SUCCESS;
}

/*-----------------------------------------------------------------------------------------------*/
