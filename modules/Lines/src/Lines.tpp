// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <iostream>
#include <math.h>
#include <omp.h>
#include <vector>
using namespace std;
#include <Eigen/Core>
using namespace Eigen;

#include "Lines.hpp"
#include "levels.hpp"
#include "linedata.hpp"
#include "acceleration_Ng.hpp"
#include "RadiativeTransfer/src/species.hpp"
#include "RadiativeTransfer/src/temperature.hpp"
#include "RadiativeTransfer/src/RadiativeTransfer.hpp"
#include "RadiativeTransfer/src/cells.hpp"
#include "RadiativeTransfer/src/lines.hpp"
#include "RadiativeTransfer/src/radiation.hpp"
#include "RadiativeTransfer/src/frequencies.hpp"


#define MAX_NITERATIONS 100


///  Lines: iteratively calculates level populations
////////////////////////////////////////////////////

template <int Dimension, long Nrays>
int Lines (CELLS<Dimension, Nrays>& cells, LINEDATA& linedata, SPECIES& species,
		       TEMPERATURE& temperature, FREQUENCIES& frequencies, LEVELS& levels,
					 RADIATION& radiation)
{

	// Initialize levels with LTE populations

	levels.set_LTE_populations (linedata, species, temperature);


	long nfreq_scat = 1;

	LINES lines (cells.ncells, linedata);

	SCATTERING scattering (Nrays, nfreq_scat);


  int niterations = 0;   // number of iterations


  // Iterate as long as some levels are not converged

  while (levels.some_not_converged)
  {
		niterations++;


		// Print number of current iteration

		cout << "(Lines): Level populations iteration " << niterations << endl;


    // Perform an Ng acceleration step every 4th iteration

    if (niterations%4 == 0)
    {
      levels.update_using_Ng_acceleration ();
    }


		// Update previous populations, making memory available for the new ones

    levels.update_previous_populations ();


    // Calculate source and opacity for all transitions over whole grid

		//TODO Maybe move this function to levels?

	  lines.get_emissivity_and_opacity (linedata, levels);



		vector<vector<double>> J (levels.ncells, vector<double> (frequencies.nfreq));

		long rays[Nrays];

		for (long r = 0; r < Nrays; r++)
		{
			rays[r] = r;
		}

		// Get radiation field from Radiative Transfer

		cout << "In RT..." << endl;
    RadiativeTransfer<Dimension, Nrays> (cells, temperature, frequencies, Nrays, rays, lines, scattering, radiation, J);
		cout << "Out RT..." << endl;
 

#   pragma omp parallel                                                      \
    shared (linedata, lines, levels, species, frequencies, temperature, J)   \
    default (none)
    {

    const int num_threads = omp_get_num_threads();
    const int thread_num  = omp_get_thread_num();

    const long start = (thread_num*levels.ncells)/num_threads;
    const long stop  = ((thread_num+1)*levels.ncells)/num_threads;   // Note brackets


    for (long p = start; p < stop; p++)
    {

      // For each species producing lines

      for (int l = 0; l < linedata.nlspec; l++)
      {

        levels.calc_J_eff (temperature, J, p, l, k);

    		MatrixXd R = linedata.calc_transition_matrix (species, temperature_gas, levels.J_eff, p, l));
    	
        levels.update_using_statistical_equilibrium (R, p, l);

    		levels.check_for_convergence (p, l);


      } // end of lspec loop over line producing species

    } // end of n loop over cells
    } // end of OpenMP parallel region



		
    // Allow 1% to be not converged

    for (int l = 0; l < linedata.nlspec; l++)
    {
      if (levels.fraction_not_converged[l] < 0.01)
      {
        levels.not_converged[l] = false;
      }
    }


    // If some are not converged

    some_not_converged = false;

    for (int l = 0; l < linedata.nlspec; l++)
    {
      if (not_converged[l])
      {
        some_not_converged = true;
      }
    }


		// Limit the number of iteration

    if (niterations > MAX_NITERATIONS)
		{
			some_not_converged = false;
		}


		// Print status of convergence

    for (int l = 0; l < linedata.nlspec; l++)
    {
			cout << "(Lines): fraction_not_converged = " << levels.fraction_not_converged << endl;
    }


  } // end of while loop of iterations



  // Print convergence stats

  cout << "(Lines): populations converged after " << niteration << "iterations" << endl;


  return (0);

}
