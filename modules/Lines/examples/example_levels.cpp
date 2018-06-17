// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <iostream>
#include <vector>
using namespace std;
#include <mpi.h>


#include "levels.hpp"
#include "linedata.hpp"
#include "RadiativeTransfer/src/species.hpp"
#include "RadiativeTransfer/src/temperature.hpp"
#include "RadiativeTransfer/src/lines.hpp"
#include "RadiativeTransfer/src/frequencies.hpp"



int main (void)
{

	cout << "levels example." << endl;

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
 

	MPI_Datatype MPI_LINES;
	MPI_Type_contiguous (2, MPI_DOUBLE, &MPI_LINES);
  MPI_Type_commit (&MPI_LINES);

  // Print off a hello world message
  cout << "Hello world from processor "<< processor_name
 	     << " rank "                     << world_rank
 	  	 << " out of "                   << world_size
 	 	 	 << " processors."               << endl;
 

	const long ncells    = 12;
	const long nspec     = 5;

	string   species_file = "/home/frederik/Dropbox/Astro/Magritte/modules/RadiativeTransfer/tests/test_data/species.txt";
  string abundance_file = "/home/frederik/Dropbox/Astro/Magritte/modules/RadiativeTransfer/tests/test_data/abundance.txt";



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


	LINES lines (ncells, linedata);

	LEVELS levels (ncells, linedata);

	levels.iteration_using_LTE (linedata, species, temperature, lines);


  for (long p = 0; p < ncells; p++)
  {
    temperature.gas[p] = 10.0;
  }

  // Finalize the MPI environment.
  MPI_Finalize ();	


  cout << "Done." << endl;



	return (0);

}
