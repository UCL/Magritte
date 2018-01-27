// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <stdio.h>
#include <math.h>
#include <omp.h>

#include "../parameters.hpp"
#include "Magritte_config.hpp"
#include "declarations.hpp"

#include "calc_LTE_populations.hpp"


// calc_LTE_populations: Calculates LTE level populations
// ------------------------------------------------------

int calc_LTE_populations (long ncells, CELL *cell, LINE_SPECIES line_species, double *pop)
{


  // For each line producing species at each grid point

  for (int lspec = 0; lspec < NLSPEC; lspec++)
  {

#   pragma omp parallel                                               \
    shared (ncells, cell, line_species, pop, nlev, cum_nlev, lspec)   \
    default (none)
    {

    int num_threads = omp_get_num_threads();
    int thread_num  = omp_get_thread_num();

    long start = (thread_num*NCELLS)/num_threads;
    long stop  = ((thread_num+1)*NCELLS)/num_threads;   // Note brackets


    for (long n = start; n < stop; n++)
    {

      // Calculate partition function

      double partition_function = 0.0;
      double total_population   = 0.0;

      for (int i = 0; i < nlev[lspec]; i++)
      {
        int l_i = LSPECLEV(lspec,i);

        partition_function = partition_function
                             + line_species.weight[l_i]
                               * exp( -line_species.energy[l_i] / (KB*cell[n].temperature.gas) );
      } // end of i loop over levels


      // Calculate LTE level populations

      for (int i = 0; i < nlev[lspec]; i++)
      {
        long p_i = LSPECGRIDLEV(lspec,n,i);
        int  l_i = LSPECLEV(lspec,i);

        pop[p_i] = cell[n].density * cell[n].abundance[line_species.nr[lspec]] * line_species.weight[l_i]
                   * exp( -line_species.energy[l_i]/(KB*cell[n].temperature.gas) ) / partition_function;

        total_population = total_population + pop[p_i];


        // Avoid too small numbers

        // if (pop[p_i] < POP_LOWER_LIMIT){
        //
        //   pop[p_i] = 0.0;
        // }

      } // end of i loop over levels

      // Check if total population adds up to density

      if ((total_population-cell[n].density*cell[n].abundance[line_species.nr[lspec]])/total_population > 1.0E-3)
      {
        printf ("\nERROR : total of level populations differs from density !\n\n");
      }

    } // end of n loop over cells
    } // end of OpenMP parallel region

  } // end of lspec loop over line producin species


  return (0);

}
