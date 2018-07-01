// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <iostream>
#include <vector>
using namespace std;
#include <mpi.h>


#include "Lines.hpp"
#include "levels.hpp"
#include "linedata.hpp"
#include "RadiativeTransfer/src/species.hpp"
#include "RadiativeTransfer/src/temperature.hpp"
#include "RadiativeTransfer/src/RadiativeTransfer.hpp"
#include "RadiativeTransfer/src/cells.hpp"
#include "RadiativeTransfer/src/lines.hpp"
#include "RadiativeTransfer/src/radiation.hpp"
#include "RadiativeTransfer/src/frequencies.hpp"



int main (void)
{


	MPI_Init (NULL, NULL);



	const int  Dimension = 1;
	const long ncells    = 50;
	const long Nrays     = 2;
	const long nspec     = 5;


	const string project_folder = "/home/frederik/Dropbox/Astro/Magritte/modules/RadiativeTransfer/tests/test_data/"

	const string       cells_file = project_folder + "grid.txt";
	const string     species_file = project_folder + "species.txt";
  const string   abundance_file = project_folder + "abundance.txt";
  const string temperature_file = project_folder + "temperature.txt";

  // long nrays = Nrays/(2*world_size);

	CELLS<Dimension, Nrays> cells (ncells);

	cells.read (cells_file);


	LINEDATA linedata;   // object containing line data


	SPECIES species (ncells, nspec, species_file);

	species.read (abundance_file);


	TEMPERATURE temperature (ncells);

	temperature.read (temperature_file);


	FREQUENCIES frequencies (ncells, linedata);

	frequencies.reset (linedata, temperature);

	const long nfreq_red = frequencies.nfreq_red;

	LEVELS levels (ncells, linedata);

  int world_size;
  MPI_Comm_size (MPI_COMM_WORLD, &world_size);

  int world_rank;
  MPI_Comm_rank (MPI_COMM_WORLD, &world_rank);

  const long START_raypair = ( world_rank   *Nrays/2)/world_size;
  const long STOP_raypair  = ((world_rank+1)*Nrays/2)/world_size;

	const long nrays_red = STOP_raypair - START_raypair;

	RADIATION radiation (ncells, nrays_red, nfreq_red, START_raypair);
	

 // for (long r = 0; r < Nrays; r++)
 // {
 // 	for (long p = 0; p < ncells; p++)
 // 	{
 // 		for (long f = 0; f < nfreq; f++)
 // 		{
 // 		  cout << radiation.V[r][p][f] << endl;  	
 // 		}
 // 	}
 // }


	Lines<Dimension, Nrays> (cells, linedata, species, temperature, frequencies, levels, radiation);


  MPI_Finalize ();	


	return (0);

}
