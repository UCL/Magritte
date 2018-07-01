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
#include "RadiativeTransfer/src/lines.hpp"
#include "RadiativeTransfer/src/temperature.hpp"
#include "RadiativeTransfer/src/frequencies.hpp"


int LEVELS ::
    iteration_using_LTE (const LINEDATA& linedata, const SPECIES& species,
	  	                   const TEMPERATURE& temperature, LINES& lines)
{


	int world_size;
	MPI_Comm_size (MPI_COMM_WORLD, &world_size);

	int world_rank;
	MPI_Comm_rank (MPI_COMM_WORLD, &world_rank);

	const long START = ( world_rank   *ncells)/world_size;
	const long STOP  = ((world_rank+1)*ncells)/world_size;

	const long ncells_red = STOP - START;


# pragma omp parallel                              \
  shared (linedata, species, temperature, lines)   \
  default (none)
  {

    const int nthreads = omp_get_num_threads();
    const int thread   = omp_get_thread_num();

    const long start = START + ( thread   *ncells_red)/nthreads;
    const long stop  = START + ((thread+1)*ncells_red)/nthreads;


	  // For all cells

    for (long p = start; p < stop; p++)
    {

      // For each species producing lines

      for (int l = 0; l < nlspec; l++)
      {

	      // Initialize levels with LTE populations

	      update_using_LTE (linedata, species, temperature, p, l);


        // Calculate line source and opacity for the new levels

	      calc_line_emissivity_and_opacity (linedata, lines, p, l);

//				lines.emissivity[p][0][0] = p;
//				lines.emissivity_vec[lines.index(p,0,0)] = p;
	  	}
	  }

	} // end of pragma omp parallel


	// Gather emissivities and opacities from all processes

	lines.mpi_allgatherv ();


//	if (world_rank == 0)
//  {
//    for (long p = 0; p < ncells; p++)
//    {
//    	cout << "op " << p << " " << lines.opacity_vec[lines.index(p,0,0)] << endl;
//    }
//  }


	return (0);

}




int LEVELS ::
    iteration_using_statistical_equilibrium (const LINEDATA& linedata,
				                                     const SPECIES& species,
	  	                                       const TEMPERATURE& temperature,
																						 FREQUENCIES& frequencies,
																						 RADIATION& radiation,
																						 LINES& lines)
{


	int world_size;
	MPI_Comm_size (MPI_COMM_WORLD, &world_size);

	int world_rank;
	MPI_Comm_rank (MPI_COMM_WORLD, &world_rank);

	const long START = ( world_rank   *ncells)/world_size;
	const long STOP  = ((world_rank+1)*ncells)/world_size;

	const long ncells_red = STOP - START;


# pragma omp parallel                                                      \
  shared (linedata, species, temperature, lines, frequencies, radiation)   \
  default (none)
  {

    const int nthreads = omp_get_num_threads();
    const int thread   = omp_get_thread_num();

    const long start = START + ( thread   *ncells_red)/nthreads;
    const long stop  = START + ((thread+1)*ncells_red)/nthreads;


	  // For all cells

    for (long p = start; p < stop; p++)
    {

      // For each species producing lines

      for (int l = 0; l < nlspec; l++)
      {

		    // Update previous populations, making memory available for the new ones

        population_prev3[p][l] = population_prev2[p][l];
        population_prev2[p][l] = population_prev1[p][l];
        population_prev1[p][l] = population[p][l];


				// Extract the effective mean radiation field in each line

        calc_J_eff (frequencies, temperature, radiation, p, l);


				// Calculate the transition matrix (for stat. equil. eq.)

    		MatrixXd R = linedata.calc_transition_matrix (species, temperature.gas[p], J_eff, p, l);
    	
        update_using_statistical_equilibrium (R, p, l);

    		check_for_convergence (p, l);


        // Calculate source and opacity

	      calc_line_emissivity_and_opacity (linedata, lines, p, l);
	  	}
	  }

	} // end of pragma omp parallel


	// Gather emissivities and opacities from all processes

	lines.mpi_allgatherv ();


//	if (world_rank == 0)
//  {
//    for (long p = 0; p < ncells; p++)
//    {
//    	cout << "op " << p << " " << lines.opacity_vec[lines.index(p,0,0)] << endl;
//    }
//  }


	return (0);

}
