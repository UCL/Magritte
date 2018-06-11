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

	cout << "Lines example." << endl;

//  // Initialize MPI environment
//	MPI_Init (NULL, NULL);
//
//  // Get the number of processes
//  int world_size;
//  MPI_Comm_size (MPI_COMM_WORLD, &world_size);
//
//  // Get the rank of the process
//  int world_rank;
//  MPI_Comm_rank (MPI_COMM_WORLD, &world_rank);
//
//
//  // Get the name of the processor
//  char processor_name[MPI_MAX_PROCESSOR_NAME];
//  int  name_len;
//  MPI_Get_processor_name (processor_name, &name_len);
// 
//
//  // Print off a hello world message
//  cout << "Hello world from processor "<< processor_name
// 	     << " rank "                     << world_rank
// 	  	 << " out of "                   << world_size
// 	 	 	 << " processors."               << endl;
// 

	const int  Dimension = 1;
	const long ncells    = 50;
	const long Nrays     = 2;
	const long nspec     = 5;

	string     cells_file = "/home/frederik/Dropbox/Astro/Magritte/modules/RadiativeTransfer/tests/test_data/grid.txt";
	string   species_file = "/home/frederik/Dropbox/Astro/Magritte/modules/RadiativeTransfer/tests/test_data/species.txt";
  string abundance_file = "/home/frederik/Dropbox/Astro/Magritte/modules/RadiativeTransfer/tests/test_data/abundance.txt";

  // long nrays = Nrays/(2*world_size);

	CELLS<Dimension, Nrays> cells (ncells);

	cells.read (cells_file);

	cells.boundary[0]        = true;
	cells.boundary[ncells-1] = true;
	
	cells.neighbor[0][0]        = 1;
	cells.n_neighbors[0]        = 1;
	cells.neighbor[ncells-1][0] = ncells-2;
	cells.n_neighbors[ncells-1] = 1;


  for (long p = 1; p < ncells-1; p++)
  {
     cells.neighbor[p][0] = p-1;
     cells.neighbor[p][1] = p+1;
	   cells.n_neighbors[p] = 2;
  }


	LINEDATA linedata;   // object containing line data


	SPECIES species (ncells, nspec, species_file);

	species.read (abundance_file);


	TEMPERATURE temperature (ncells);

  for (long p = 0; p < ncells; p++)
  {
    temperature.gas[p] = 10.0;
  }

	FREQUENCIES frequencies (ncells, linedata);

	frequencies.reset (linedata, temperature);

	const long nfreq = frequencies.nfreq;

	LEVELS levels (ncells, linedata);

	RADIATION radiation (ncells, Nrays, nfreq);
	

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


  // Finalize the MPI environment.
  // MPI_Finalize ();	


  cout << "Done." << endl;


	return (0);

}
