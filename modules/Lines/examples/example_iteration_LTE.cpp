// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <iostream>
#include <fstream>
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


  // Initialize MPI environment
	MPI_Init (NULL, NULL);

  // Get the number of processes
  int world_size;
  MPI_Comm_size (MPI_COMM_WORLD, &world_size);

  // Get the rank of the process
  int world_rank;
  MPI_Comm_rank (MPI_COMM_WORLD, &world_rank);


	const int  Dimension = 3;
	const long ncells    = 5*5*5;
	const long Nrays     = 12;
	const long nspec     = 5;


	const string project_folder = "example_data/iteration_LTE/";
	const string   input_folder = project_folder + "input/";
	const string  output_folder = project_folder + "output/";

	const string       cells_file = input_folder + "grid.txt";
	const string     species_file = input_folder + "species.txt";
  const string   abundance_file = input_folder + "abundance.txt";
  const string temperature_file = input_folder + "temperature.txt";


	LINEDATA linedata;


	SPECIES species (ncells, nspec, species_file);

	species.read (abundance_file);


	TEMPERATURE temperature (ncells);

	temperature.read (temperature_file);


	LINES lines (ncells, linedata);

	LEVELS levels (ncells, linedata);


	TIMER timer;	
	timer.start();

	levels.iteration_using_LTE (linedata, species, temperature, lines);

	timer.stop();
	timer.print();



	// Print out results

	const string output_file = output_folder + "populations.txt";  
  ofstream outputFile (output_file);
	if (outputFile.is_open())
  {
	 int l = 0;
	 for (long p = 0; p < ncells; p++)
	 {
		 for (int i = 0; i < levels.nlev[l]; i++)
		 {
			 outputFile << levels.population[p][l][i] << "\t";
		 }
	 	 outputFile << "\n";
	 }

	  outputFile.close();
  }
  else cout << "Unable to open file";


  // Finalize the MPI environment.
  MPI_Finalize ();	


	return (0);

}
