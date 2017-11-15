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

int thermal_balance_iteration( GRIDPOINT *gridpoint, EVALPOINT *evalpoint, long *antipod,
                               double *column_H2, double *column_HD, double *column_C,
                               double *column_CO, double *UV_field,
                               double *temperature_gas, double *temperature_dust,
                               double *rad_surface, double *AV, int *irad, int *jrad,
                               double *energy, double *weight, double *frequency,
                               double *A_coeff, double *B_coeff, double *C_coeff, double *R,
                               double *C_data, double *coltemp, int *icol, int *jcol,
                               double *prev1_pop, double *prev2_pop, double *prev3_pop,
                               double *pop, double *mean_intensity,
                               double *Lambda_diagonal, double *mean_intensity_eff,
                               double *thermal_ratio,
                               double *time_chemistry, double *time_level_pop )
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

    chemistry( gridpoint, evalpoint, temperature_gas, temperature_dust, rad_surface, AV,
               column_H2, column_HD, column_C, column_CO );

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

  level_populations( gridpoint, evalpoint, antipod, irad, jrad, frequency,
                     A_coeff, B_coeff, C_coeff, R, pop, prev1_pop, prev2_pop, prev3_pop,
                     C_data, coltemp, icol, jcol, temperature_gas, temperature_dust,
                     weight, energy, mean_intensity, Lambda_diagonal, mean_intensity_eff );

  *time_level_pop += omp_get_wtime();


  printf("\n(thermal_balance): time in level_populations: %lf sec\n", *time_level_pop);


  printf("(thermal_balance): level populations calculated \n\n");


  /*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/




  /*   CALCULATE HEATING AND COOLING                                                             */
  /*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


  printf("(thermal_balance): calculating heating and cooling \n\n");


  /* Calculate column densities to get the most recent reaction rates */

  calc_column_density(gridpoint, evalpoint, column_H2, H2_nr);
  calc_column_density(gridpoint, evalpoint, column_HD, HD_nr);
  calc_column_density(gridpoint, evalpoint, column_C, C_nr);
  calc_column_density(gridpoint, evalpoint, column_CO, CO_nr);


  /* Calculate the thermal balance for each gridpoint */

  for (long gridp=0; gridp<NGRID; gridp++){

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


  printf("(thermal_balance): heating and cooling calculated \n\n");


  /*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/




  return(0);

}

/*-----------------------------------------------------------------------------------------------*/