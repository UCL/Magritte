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

  // Initialize MPI environment
	MPI_Init (NULL, NULL);

  // Get the number of processes
  int world_size;
  MPI_Comm_size (MPI_COMM_WORLD, &world_size);

  // Get the rank of the process
  int world_rank;
  MPI_Comm_rank (MPI_COMM_WORLD, &world_rank);


  // Get the name of the processor
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int  name_len;
  MPI_Get_processor_name (processor_name, &name_len);
 

  // Print off a hello world message
  cout << "Hello world from processor "<< processor_name
 	     << " rank "                     << world_rank
 	  	 << " out of "                   << world_size
 	 	 	 << " processors."               << endl;
 

	const int  Dimension = 1;
	const long ncells    = 10;
	const long Nrays     = 2;

  // long nrays = Nrays/(2*world_size);

	CELLS<Dimension, Nrays> cells (ncells);

	LINEDATA linedata;   // object containing line data

	string species_file = "/home/frederik/Dropbox/Astro/MagritteProjects/3DPDR/data/species_reduced.txt";

	SPECIES species (ncells, 35, species_file);

	TEMPERATURE temperature (ncells);

	FREQUENCIES frequencies (ncells, linedata);

	long nfreq = frequencies.nfreq;

	LEVELS levels (ncells, linedata);

	RADIATION radiation (ncells, Nrays, nfreq);
	
	Lines<Dimension, Nrays> (cells, linedata, species, temperature, frequencies, levels, radiation);


  // Finalize the MPI environment.
  MPI_Finalize ();	


  cout << "Done." << endl;


	return (0);

}
