// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#include "../parameters.hpp"
#include "Magritte_config.hpp"
#include "declarations.hpp"

#include "thermal_balance.hpp"
#include "initializers.hpp"
#include "calc_column_density.hpp"
#include "chemistry.hpp"
#include "calc_LTE_populations.hpp"
#include "level_populations_otf.hpp"
#include "cell_level_populations.hpp"
#include "reaction_rates.hpp"
#include "heating.hpp"
#include "cooling.hpp"


// thermal_balance: perform a thermal balance iteration to calculate thermal flux
// ------------------------------------------------------------------------------

int thermal_balance (long ncells, CELL *cell, SPECIES *species, REACTION *reaction,
                     double *column_H2, double *column_HD, double *column_C, double *column_CO,
                     double *UV_field, double *rad_surface, double *AV, int *irad, int *jrad,
                     double *energy, double *weight, double *frequency, double *A_coeff, double *B_coeff,
                     double *C_data, double *coltemp, int *icol, int *jcol, double *pop,
                     double *mean_intensity, double *Lambda_diagonal, double *mean_intensity_eff,
                     double *thermal_ratio, double *time_chemistry, double *time_level_pop)
{

  // CALCULATE CHEMICAL ABUNDANCES
  // +++++++++++++++++++++++++++++


  printf("(thermal_balance): calculating chemical abundances \n\n");


# if (ALWAYS_INITIALIZE_CHEMISTRY)

    initialize_abundances (NCELLS, cell, species);

# endif


  // Calculate chemical abundances by solving rate equations

  for (int chem_iteration = 0; chem_iteration < CHEM_ITER; chem_iteration++)
  {
    printf( "(thermal_balance):   chemistry iteration %d of %d \n",
            chem_iteration+1, CHEM_ITER );


    // Calculate chemical abundances given current temperatures and radiation field

    *time_chemistry -= omp_get_wtime();


    chemistry (NCELLS, cell, species, reaction, rad_surface, AV, column_H2, column_HD, column_C, column_CO );


    *time_chemistry += omp_get_wtime();


  } // End of chemistry iteration


  printf("\n(thermal_balance): time in chemistry: %lf sec\n", *time_chemistry);

  printf("(thermal_balance): chemical abundances calculated \n\n");




  // CALCULATE LEVEL POPULATIONS (ITERATIVELY)
  // +++++++++++++++++++++++++++++++++++++++++


  printf("(thermal_balance): calculating level populations \n\n");


  // Initialize level populations with LTE values

  calc_LTE_populations (NCELLS, cell, energy, weight, pop);


  // Calculate level populations for each line producing species

  *time_level_pop -= omp_get_wtime();


# if (CELL_BASED)

  cell_level_populations (NCELLS, cell, irad, jrad, frequency, A_coeff, B_coeff, pop, C_data, coltemp,
                          icol, jcol, weight, energy, mean_intensity, Lambda_diagonal, mean_intensity_eff);

# else

  level_populations_otf (NCELLS, cell, irad, jrad, frequency, A_coeff, B_coeff, pop, C_data, coltemp,
                         icol, jcol, weight, energy, mean_intensity, Lambda_diagonal, mean_intensity_eff);

# endif


  *time_level_pop += omp_get_wtime();


  printf("\n(thermal_balance): time in level_populations: %lf sec\n", *time_level_pop);


  printf("(thermal_balance): level populations calculated \n\n");




  // CALCULATE HEATING AND COOLING
  // +++++++++++++++++++++++++++++


  printf("(thermal_balance): calculating heating and cooling \n\n");


  // Calculate column densities to get most recent reaction rates

  calc_column_densities (NCELLS, cell, column_H2, column_HD, column_C, column_CO);


  // Calculate thermal balance for each cell

# pragma omp parallel                                                                                \
  shared (ncells, cell, reaction, irad, jrad, A_coeff, B_coeff, pop, frequency, weight, column_H2,   \
          column_HD, column_C, column_CO, cum_nlev, species, mean_intensity, AV, rad_surface,        \
          UV_field, thermal_ratio)                                                                   \
  default (none)
  {

  int num_threads = omp_get_num_threads();
  int thread_num  = omp_get_thread_num();

  long start = (thread_num*NCELLS)/num_threads;
  long stop  = ((thread_num+1)*NCELLS)/num_threads;   // Note brackets


  for (long gridp = start; gridp < stop; gridp++)
  {
    double heating_components[12];


    reaction_rates (NCELLS, cell, reaction, gridp, rad_surface, AV, column_H2, column_HD, column_C, column_CO);


    double heating_total = heating (NCELLS, cell, gridp, UV_field, heating_components);

    double cooling_total = cooling (NCELLS, gridp, irad, jrad, A_coeff, B_coeff, frequency, weight,
                                    pop, mean_intensity);


    double thermal_flux = heating_total - cooling_total;

    double thermal_sum  = heating_total + cooling_total;


    thermal_ratio[gridp] = 0.0;

    if (thermal_sum != 0.0)
    {
      thermal_ratio[gridp] = 2.0 * thermal_flux / thermal_sum;
    }

  } // end of gridp loop over cells
  } // end of OpenMP parallel region


  printf("(thermal_balance): heating and cooling calculated \n\n");


  return(0);

}
