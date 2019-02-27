// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <math.h>
#include <mpi.h>
#include <omp.h>
#include <iostream>
using namespace std;

#include "levels.hpp"
#include "linedata.hpp"
#include "RadiativeTransfer/src/constants.hpp"
#include "RadiativeTransfer/src/ompTools.hpp"
#include "RadiativeTransfer/src/mpiTools.hpp"
#include "RadiativeTransfer/src/hybridTools.hpp"
#include "RadiativeTransfer/src/lines.hpp"
#include "RadiativeTransfer/src/temperature.hpp"
#include "RadiativeTransfer/src/frequencies.hpp"


int LEVELS ::
    iteration_using_LTE (const LINEDATA    &linedata,
                         const SPECIES     &species,
	  	         const TEMPERATURE &temperature,
                               LINES       &lines       )
{

# pragma omp parallel                              \
  shared (linedata, species, temperature, lines)   \
  default (none)
  {

    // For all cells

    for (long p = HYBRID_start (ncells); p < HYBRID_stop (ncells); p++)
    {

      // For each species producing lines

      for (int l = 0; l < nlspec; l++)
      {
        // Initialize levels with LTE populations

        update_using_LTE (linedata, species, temperature, p, l);


        // Calculate line source and opacity for the new levels

        calc_line_emissivity_and_opacity (linedata, lines, p, l);
      }
    }

  } // end of pragma omp parallel


  // Gather emissivities and opacities from all processes
  // (since we used a HYBRID loop)

  lines.mpi_allgatherv ();


  return (0);

}




int LEVELS ::
    iteration_using_statistical_equilibrium (const LINEDATA    &linedata,
                                             const SPECIES     &species,
	  	                             const TEMPERATURE &temperature,
                                             const FREQUENCIES &frequencies,
                                             const RADIATION   &radiation,
                                                   LINES       &lines       )
{

# pragma omp parallel                                                      \
  shared (linedata, species, temperature, lines, frequencies, radiation)   \
  default (none)
  {

    // For all cells

    for (long p = HYBRID_start (ncells); p < HYBRID_stop (ncells); p++)
    {

      // For each species producing lines

      for (int l = 0; l < nlspec; l++)
      {

        // Update previous populations, making memory available for the new ones

        population_prev3[p][l] = population_prev2[p][l];
        population_prev2[p][l] = population_prev1[p][l];
        population_prev1[p][l] = population[p][l];


        // Extract the effective mean radiation field in each line

        calc_J_and_L_eff (frequencies, temperature, radiation, p, l);


        // Calculate the transition matrix (for stat. equil. eq.)

        MatrixXd R = linedata.calc_transition_matrix (species, /*lines,*/ temperature.gas[p], J_eff, p, l);


        // Update levels

        update_using_statistical_equilibrium (R, p, l);


        // Check for convergence

        check_for_convergence (p, l);


        // Calculate source and opacity

        calc_line_emissivity_and_opacity (linedata, lines, p, l);
      }
    }

  } // end of pragma omp parallel


  // Gather emissivities and opacities from all processes
  // (since we used a HYBRID loop)

  lines.mpi_allgatherv ();


  return (0);

}
