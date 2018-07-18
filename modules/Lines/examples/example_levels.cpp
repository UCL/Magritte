// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <iostream>
#include <vector>
using namespace std;
#include <mpi.h>

#include "RadiativeTransfer/src/timer.hpp"

#include "levels.hpp"
#include "linedata.hpp"
#include "RadiativeTransfer/src/species.hpp"
#include "RadiativeTransfer/src/temperature.hpp"
#include "RadiativeTransfer/src/lines.hpp"
#include "RadiativeTransfer/src/frequencies.hpp"


int main (void)
{

	MPI_TIMER timer ('levels');	
	timer.start();

  // Initialize MPI environment
	MPI_Init (NULL, NULL);

  // Get the number of processes
  int world_size;
  MPI_Comm_size (MPI_COMM_WORLD, &world_size);

  // Get the rank of the process
  int world_rank;
  MPI_Comm_rank (MPI_COMM_WORLD, &world_rank);


	const int  Dimension = 1;
	const long ncells    = 50;
	const long Nrays     = 2;
	const long nspec     = 5;


	const string project_folder = "/home/frederik/Dropbox/Astro/Magritte/modules/RadiativeTransfer/tests/test_data/";

	const string       cells_file = project_folder + "grid.txt";
	const string     species_file = project_folder + "species.txt";
  const string   abundance_file = project_folder + "abundance.txt";
  const string temperature_file = project_folder + "temperature.txt";


	LINEDATA linedata;


	SPECIES species (ncells, nspec, species_file);

	species.read (abundance_file);


	TEMPERATURE temperature (ncells);

	temperature.read (temperature_file);


//	FREQUENCIES frequencies (ncells, linedata);
//
//  frequencies.reset (linedata, temperature);
//
//	const long nfreq_red = frequencies.nfreq_red;


	LINES lines (ncells, linedata);

	LEVELS levels (ncells, linedata);



	levels.iteration_using_LTE (linedata, species, temperature, lines);



  // Finalize the MPI environment.
  MPI_Finalize ();

	timer.stop();
	timer.print();

	return (0);

}
