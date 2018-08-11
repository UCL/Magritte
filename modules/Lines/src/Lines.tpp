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
#include "RadiativeTransfer/src/timer.hpp"
#include "RadiativeTransfer/src/types.hpp"
#include "RadiativeTransfer/src/GridTypes.hpp"
#include "RadiativeTransfer/src/cells.hpp"
#include "RadiativeTransfer/src/lines.hpp"
#include "RadiativeTransfer/src/species.hpp"
#include "RadiativeTransfer/src/radiation.hpp"
#include "RadiativeTransfer/src/frequencies.hpp"
#include "RadiativeTransfer/src/temperature.hpp"
#include "RadiativeTransfer/src/RadiativeTransfer.hpp"


#define MAX_NITERATIONS 1


///  Lines: iteratively calculates level populations
////////////////////////////////////////////////////

template <int Dimension, long Nrays>
int Lines (CELLS<Dimension, Nrays>& cells, LINEDATA& linedata, SPECIES& species,
		       TEMPERATURE& temperature, FREQUENCIES& frequencies, LEVELS& levels,
					 RADIATION& radiation)
{

	const long nfreq_scat = 1;

	LINES lines (cells.ncells, linedata);

	SCATTERING scattering (Nrays, nfreq_scat, frequencies.nfreq_red);


	// Initialize levels, emissivities and opacities with LTE values

	MPI_TIMER timer_LTE ("LTE");
	timer_LTE.start ();

  levels.iteration_using_LTE (linedata, species, temperature, lines);

	timer_LTE.stop ();
	timer_LTE.print_to_file ();

  {
	#include "RadiativeTransfer/src/folders.hpp"
	levels.print (output_folder, "_LTE");
  }


  int niterations = 0;   // number of iterations


  // Iterate as long as some levels are not converged

  while (levels.some_not_converged)
  {
		niterations++;


		// Print number of current iteration

		cout << "(Lines): Starting iteration " << niterations << endl;


    // Perform an Ng acceleration step every 4th iteration

    //if (niterations%4 == 0)
    //{
    //  levels.update_using_Ng_acceleration ();


    //  // Calculate source and opacity

	  //  //levels.calc_line_emissivity_and_opacity (linedata, lines, p, l);
    //}

    {
		#include "RadiativeTransfer/src/folders.hpp"
		lines.print (output_folder, "_" + to_string(niterations));
	  }

		// Get radiation field from Radiative Transfer

		MPI_TIMER timer_RT ("RT");
		timer_RT.start ();

cout << "before RT" << endl;

    RadiativeTransfer<Dimension, Nrays>
			               (cells, temperature, frequencies, lines, scattering, radiation);

cout << "after RT" << endl;

	  timer_RT.stop ();
	  timer_RT.print_to_file ();


    for (int l = 0; l < linedata.nlspec; l++)
    {
		  levels.fraction_not_converged[l] = 0.0;
    }


		//MPI_TIMER timer_SE ("SE");
		//timer_SE.start ();

    levels.iteration_using_statistical_equilibrium (linedata, species, temperature,
				                                            frequencies, radiation, lines);
	  //timer_SE.stop ();
	  //timer_SE.print_to_file ();


		{
		#include "RadiativeTransfer/src/folders.hpp"
		levels.print (output_folder, "_" + to_string(niterations));
	  }

    // Allow 1% to be not converged

    for (int l = 0; l < linedata.nlspec; l++)
    {
      if (levels.fraction_not_converged[l] < 0.01)
      {
        levels.not_converged[l] = false;
      }
    }


    // If some are not converged

    levels.some_not_converged = false;

    for (int l = 0; l < linedata.nlspec; l++)
    {
      if (levels.not_converged[l])
      {
        levels.some_not_converged = true;
      }
    }


		// Limit the number of iteration

    if (niterations >= MAX_NITERATIONS)
		{
			levels.some_not_converged = false;
		}


		// Print status of convergence

    for (int l = 0; l < linedata.nlspec; l++)
    {
			cout << "(Lines): fraction_not_converged = ";
			cout << levels.fraction_not_converged[l];
			cout << endl;
    }


  } // end of while loop of iterations



  // Print convergence stats

  cout << "(Lines): converged after " << niterations << " iterations" << endl;


  return (0);

}
