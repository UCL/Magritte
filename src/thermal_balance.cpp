/* Frederik De Ceuster - University College London & KU Leuven                                   */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/* thermal_balance: Perform a thermal balance iteration to calculate the thermal flux            */
/*                                                                                               */
/* (NEW)                                                                                         */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/



#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#include "../parameters.hpp"
#include "Magritte_config.hpp"
#include "declarations.hpp"

#include "thermal_balance.hpp"
#include "calc_column_density.hpp"
#include "chemistry.hpp"
#include "calc_LTE_populations.hpp"
#include "level_populations.hpp"
#include "reaction_rates.hpp"
#include "heating.hpp"
#include "cooling.hpp"



/* thermal_balance: perform a thermal balance iteration to calculate the thermal flux            */
/*-----------------------------------------------------------------------------------------------*/

#ifdef ON_THE_FLY


int thermal_balance_iteration( GRIDPOINT *gridpoint,
                               double *column_H2, double *column_HD, double *column_C,
                               double *column_CO, double *UV_field,
                               double *temperature_gas, double *temperature_dust,
                               double *rad_surface, double *AV, int *irad, int *jrad,
                               double *energy, double *weight, double *frequency,
                               double *A_coeff, double *B_coeff, double *R,
                               double *C_data, double *coltemp, int *icol, int *jcol,
                               double *prev1_pop, double *prev2_pop, double *prev3_pop,
                               double *pop, double *mean_intensity,
                               double *Lambda_diagonal, double *mean_intensity_eff,
                               double *thermal_ratio,
                               double *time_chemistry, double *time_level_pop )


#else

int thermal_balance_iteration( GRIDPOINT *gridpoint, EVALPOINT *evalpoint,
                               long *key, long *raytot, long *cum_raytot,
                               double *column_H2, double *column_HD, double *column_C,
                               double *column_CO, double *UV_field,
                               double *temperature_gas, double *temperature_dust,
                               double *rad_surface, double *AV, int *irad, int *jrad,
                               double *energy, double *weight, double *frequency,
                               double *A_coeff, double *B_coeff, double *R,
                               double *C_data, double *coltemp, int *icol, int *jcol,
                               double *prev1_pop, double *prev2_pop, double *prev3_pop,
                               double *pop, double *mean_intensity,
                               double *Lambda_diagonal, double *mean_intensity_eff,
                               double *thermal_ratio,
                               double *time_chemistry, double *time_level_pop )

#endif


{




  /*   CALCULATE CHEMICAL ABUNDANCES                                                             */
  /*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


  printf("(thermal_balance): calculating chemical abundances \n\n");


  /* Calculate the chemical abundances by solving the rate equations */

  for (int chem_iteration=0; chem_iteration<CHEM_ITER; chem_iteration++){

    printf( "(thermal_balance):   chemistry iteration %d of %d \n",
            chem_iteration+1, CHEM_ITER );


    /* Calculate the chemical abundances given the current temperatures and radiation field */

    *time_chemistry -= omp_get_wtime();


#   ifdef ON_THE_FLY

    chemistry( gridpoint, temperature_gas, temperature_dust, rad_surface, AV,
               column_H2, column_HD, column_C, column_CO );

#   else

    chemistry( gridpoint, evalpoint, key, raytot, cum_raytot, temperature_gas, temperature_dust,
               rad_surface, AV, column_H2, column_HD, column_C, column_CO );

#   endif


    *time_chemistry += omp_get_wtime();


  } /* End of chemistry iteration */


  printf("\n(thermal_balance): time in chemistry: %lf sec\n", *time_chemistry);

  printf("(thermal_balance): chemical abundances calculated \n\n");


  /*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/




  /*   CALCULATE LEVEL POPULATIONS (ITERATIVELY)                                                 */
  /*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


  printf("(thermal_balance): calculating level populations \n\n");


  /* Initialize the level populations to their LTE values */

  calc_LTE_populations(gridpoint, energy, weight, temperature_gas, pop);


  /* Calculate level populations for each line producing species */

  *time_level_pop -= omp_get_wtime();


# ifdef ON_THE_FLY

  level_populations( gridpoint, irad, jrad, frequency,
                     A_coeff, B_coeff, R, pop, prev1_pop, prev2_pop, prev3_pop,
                     C_data, coltemp, icol, jcol, temperature_gas, temperature_dust,
                     weight, energy, mean_intensity, Lambda_diagonal, mean_intensity_eff );

# else

  level_populations( gridpoint, evalpoint, key, raytot, cum_raytot, irad, jrad, frequency,
                     A_coeff, B_coeff, R, pop, prev1_pop, prev2_pop, prev3_pop,
                     C_data, coltemp, icol, jcol, temperature_gas, temperature_dust,
                     weight, energy, mean_intensity, Lambda_diagonal, mean_intensity_eff );

# endif


  *time_level_pop += omp_get_wtime();


  printf("\n(thermal_balance): time in level_populations: %lf sec\n", *time_level_pop);


  printf("(thermal_balance): level populations calculated \n\n");


  /*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/




  /*   CALCULATE HEATING AND COOLING                                                             */
  /*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


  printf("(thermal_balance): calculating heating and cooling \n\n");


  /* Calculate column densities to get the most recent reaction rates */


# ifdef ON_THE_FLY

  calc_column_densities(gridpoint, column_H2, column_HD, column_C, column_CO);

# else

  calc_column_density(gridpoint, evalpoint, key, raytot, cum_raytot, column_H2, H2_nr);
  calc_column_density(gridpoint, evalpoint, key, raytot, cum_raytot, column_HD, HD_nr);
  calc_column_density(gridpoint, evalpoint, key, raytot, cum_raytot, column_C, C_nr);
  calc_column_density(gridpoint, evalpoint, key, raytot, cum_raytot, column_CO, CO_nr);

# endif


  /* Calculate the thermal balance for each gridpoint */

# pragma omp parallel                                                                             \
  shared( gridpoint, temperature_gas, temperature_dust, irad, jrad, A_coeff, B_coeff, pop,        \
          frequency, weight, column_H2, column_HD, column_C, column_CO, cum_nlev, species,        \
          mean_intensity, AV, rad_surface, UV_field, thermal_ratio )                              \
  default( none )
  {

  int num_threads = omp_get_num_threads();
  int thread_num  = omp_get_thread_num();

  long start = (thread_num*NGRID)/num_threads;
  long stop  = ((thread_num+1)*NGRID)/num_threads;            /* Note the brackets are important */


  for (long gridp=start; gridp<stop; gridp++){

    double heating_components[12];


    reaction_rates( temperature_gas, temperature_dust, rad_surface, AV,
                    column_H2, column_HD, column_C, column_CO, gridp );


    double heating_total = heating( gridpoint, gridp, temperature_gas, temperature_dust,
                                    UV_field, heating_components );

    double cooling_total = cooling( gridp, irad, jrad, A_coeff, B_coeff, frequency, weight,
                                    pop, mean_intensity );


    double thermal_flux = heating_total - cooling_total;

    double thermal_sum  = heating_total + cooling_total;


    thermal_ratio[gridp] = 0.0;

    if (thermal_sum != 0.0){

      thermal_ratio[gridp] = 2.0 * thermal_flux / thermal_sum;
    }


  } /* end of gridp loop over grid points */
  } /* end of OpenMP parallel region */


  printf("(thermal_balance): heating and cooling calculated \n\n");


  /*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/




  return(0);

}

/*-----------------------------------------------------------------------------------------------*/
