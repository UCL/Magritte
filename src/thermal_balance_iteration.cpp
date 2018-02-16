// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#include "declarations.hpp"
#include "thermal_balance_iteration.hpp"
#include "initializers.hpp"
#include "calc_column_density.hpp"
#include "chemistry.hpp"
#include "calc_LTE_populations.hpp"
#include "level_populations_otf.hpp"
#include "cell_level_populations.hpp"
#include "reaction_rates.hpp"
#include "heating.hpp"
#include "cooling.hpp"


// thermal_balance_iteration: perform a thermal balance iteration to calculate thermal flux
// ----------------------------------------------------------------------------------------

int thermal_balance_iteration (long ncells, CELL *cell, SPECIES *species, REACTION *reaction, LINE_SPECIES line_species,
                               double *column_H2, double *column_HD, double *column_C, double *column_CO,
                               double *thermal_ratio, TIMERS *timers)
{

  // CALCULATE CHEMICAL ABUNDANCES
  // +++++++++++++++++++++++++++++


  printf ("(thermal_balance_iteration): calculating chemical abundances \n\n");


# if (ALWAYS_INITIALIZE_CHEMISTRY)

    initialize_abundances (NCELLS, cell, species);

# endif


  // Calculate chemical abundances by solving rate equations

  for (int chem_iteration = 0; chem_iteration < CHEM_ITER; chem_iteration++)
  {
    printf ( "(thermal_balance_iteration):   chemistry iteration %d of %d \n",
             chem_iteration+1, CHEM_ITER );


    // Calculate chemical abundances given current temperatures and radiation field

    timers->chemistry.start();


    chemistry (NCELLS, cell, species, reaction, column_H2, column_HD, column_C, column_CO);


    timers->chemistry.stop();


  } // End of chemistry iteration


  printf ("\n(thermal_balance_iteration): time in chemistry: %lf sec\n", timers->chemistry.duration);

  printf ("(thermal_balance_iteration): chemical abundances calculated \n\n");




  // CALCULATE LEVEL POPULATIONS (ITERATIVELY)
  // +++++++++++++++++++++++++++++++++++++++++


  printf("(thermal_balance_iteration): calculating level populations \n\n");


  // Initialize level populations with LTE values

  calc_LTE_populations (NCELLS, cell, line_species);


  // Calculate level populations for each line producing species

  timers->level_pop.start();


# if (CELL_BASED)

  cell_level_populations (NCELLS, cell, line_species);

# else

  level_populations_otf (NCELLS, cell, line_species, pop, mean_intensity, Lambda_diagonal, mean_intensity_eff);

# endif


  timers->level_pop.stop();


  printf("\n(thermal_balance_iteration): time in level_populations: %lf sec\n", timers->level_pop.duration);


  printf("(thermal_balance_iteration): level populations calculated \n\n");




  // CALCULATE HEATING AND COOLING
  // +++++++++++++++++++++++++++++


  printf("(thermal_balance_iteration): calculating heating and cooling \n\n");


  // Calculate column densities to get most recent reaction rates

  calc_column_densities (NCELLS, cell, column_H2, column_HD, column_C, column_CO);


  // Calculate thermal balance for each cell

# pragma omp parallel                                                                    \
  shared (ncells, cell, reaction, column_H2, column_HD, column_C, column_CO, cum_nlev,   \
          thermal_ratio, line_species)                                                   \
  default (none)
  {

  int num_threads = omp_get_num_threads();
  int thread_num  = omp_get_thread_num();

  long start = (thread_num*NCELLS)/num_threads;
  long stop  = ((thread_num+1)*NCELLS)/num_threads;   // Note brackets


  for (long gridp = start; gridp < stop; gridp++)
  {
    double heating_components[12];


    reaction_rates (NCELLS, cell, reaction, gridp, column_H2, column_HD, column_C, column_CO);


    double heating_total = heating (NCELLS, cell, gridp, heating_components);
    double cooling_total = cooling (NCELLS, cell, line_species, gridp);

    double thermal_flux = heating_total - cooling_total;
    double thermal_sum  = heating_total + cooling_total;

    thermal_ratio[gridp] = 0.0;

    if (thermal_sum != 0.0)
    {
      thermal_ratio[gridp] = 2.0 * thermal_flux / thermal_sum;
    }

  } // end of gridp loop over cells
  } // end of OpenMP parallel region


  printf("(thermal_balance_iteration): heating and cooling calculated \n\n");


  return(0);

}
