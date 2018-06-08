// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <math.h>
#include <iostream>
#include <omp.h>

#include "levels.hpp"
#include "linedata.hpp"
#include "RadiativeTransfer/src/constants.hpp"
#include "RadiativeTransfer/src/temperature.hpp"


///  Constructor for LEVELS
///////////////////////////

LEVELS :: LEVELS (long num_of_cells, LINEDATA linedata)
{
  
	ncells = num_of_cells;

  population.resize (ncells);

  population_prev1.resize (ncells);
  population_prev2.resize (ncells);
  population_prev3.resize (ncells);

	
# pragma omp parallel   \
	shared (linedata)     \
  default (none)
  {

  int num_threads = omp_get_num_threads();
  int thread_num  = omp_get_thread_num();

  long start = (thread_num*ncells)/num_threads;
  long stop  = ((thread_num+1)*ncells)/num_threads;   // Note brackets


  for (long p = start; p < stop; p++)
  {
    population[p].resize (linedata.nlspec);

    population_prev1[p].resize (linedata.nlspec);
    population_prev2[p].resize (linedata.nlspec);
    population_prev3[p].resize (linedata.nlspec);


		for (int l = 0; l < linedata.nlspec; l++)
		{
			population[p][l].resize (linedata.nlev[l]);

			population_prev1[p][l].resize (linedata.nlev[l]);
			population_prev2[p][l].resize (linedata.nlev[l]);
			population_prev3[p][l].resize (linedata.nlev[l]);
		}
	}

	} // end of pragma omp parallel


}   // END OF CONSTRUCTOR





int LEVELS :: set_LTE_populations (LINEDATA linedata, SPECIES species, TEMPERATURE temperature)
{

  // For each line producing species at each grid point

  for (int l = 0; l < linedata.nlspec; l++)
  {

#   pragma omp parallel                          \
    shared (linedata, species, temperature, l)   \
    default (none)
    {

    int num_threads = omp_get_num_threads();
    int thread_num  = omp_get_thread_num();

    long start = (thread_num*ncells)/num_threads;
    long stop  = ((thread_num+1)*ncells)/num_threads;   // Note brackets


    for (long p = start; p < stop; p++)
    {

      // Calculate partition function

      double partition_function = 0.0;

      for (int i = 0; i < linedata.nlev[l]; i++)
      {
        partition_function += linedata.weight[l](i) * exp( -linedata.energy[l](i) / (KB*temperature.gas[p]) );
      }


      // Calculate LTE level populations

      for (int i = 0; i < linedata.nlev[l]; i++)
      {
        population[p][l](i) = species.density[p] * species.abundance[p][linedata.num[l]] * linedata.weight[l](i)
                              * exp( -linedata.energy[l](i)/(KB*temperature.gas[p]) ) / partition_function;
      }

    } // end of n loop over cells
    } // end of OpenMP parallel region

  } // end of lspec loop over line producin species


  return (0);

}
