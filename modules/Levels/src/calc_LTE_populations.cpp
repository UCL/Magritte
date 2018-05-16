// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <stdio.h>
#include <math.h>
#include <omp.h>

#include "declarations.hpp"
#include "calc_LTE_populations.hpp"


// calc_LTE_populations: Calculates LTE level populations
// ------------------------------------------------------

int calc_LTE_populations (CELLS *cells, LINES lines)
{

  // For each line producing species at each grid point

  for (int ls = 0; ls < NLSPEC; ls++)
  {

#   pragma omp parallel                         \
    shared (cells, lines, nlev, cum_nlev, ls)   \
    default (none)
    {

    int num_threads = omp_get_num_threads();
    int thread_num  = omp_get_thread_num();

    long start = (thread_num*NCELLS)/num_threads;
    long stop  = ((thread_num+1)*NCELLS)/num_threads;   // Note brackets


    for (long p = start; p < stop; p++)
    {

      // Calculate partition function

      double partition_function = 0.0;
      double total_population   = 0.0;

      for (int i = 0; i < nlev[ls]; i++)
      {
        int l_i = LSPECLEV(ls,i);

        partition_function = partition_function
                             + lines.weight[l_i]
                               * exp( -lines.energy[l_i] / (KB*cells->temperature_gas[p]) );
      }


      // Calculate LTE level populations

      for (int i = 0; i < nlev[ls]; i++)
      {
        int l_i = LSPECLEV(ls,i);

        cells->pop[LINDEX(p,l_i)] = cells->density[p] * cells->abundance[SINDEX(p,lines.nr[ls])] * lines.weight[l_i]
                                   * exp( -lines.energy[l_i]/(KB*cells->temperature_gas[p]) ) / partition_function;

        total_population = total_population + cells->pop[LINDEX(p,l_i)];
      }


      // Check if total population adds up to density

      if ((total_population-cells->density[p]*cells->abundance[SINDEX(p,lines.nr[ls])])/total_population > 1.0E-3)
      {
        printf ("\nERROR : total of level populations differs from density !\n\n");
      }

    } // end of n loop over cells
    } // end of OpenMP parallel region

  } // end of lspec loop over line producin species


  return (0);

}
