// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <math.h>
#include <mpi.h>
#include <omp.h>
#include <iostream>
using namespace std;
#include <Eigen/QR>
using namespace Eigen;

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

	long *test;
  test = new long[ncells];

	int world_size;
	MPI_Comm_size (MPI_COMM_WORLD, &world_size);

	int world_rank;
	MPI_Comm_rank (MPI_COMM_WORLD, &world_rank);

	const long START = ( world_rank   *ncells)/world_size;
	const long STOP  = ((world_rank+1)*ncells)/world_size;

	const long ncells_red = STOP - START;


# pragma omp parallel                              \
  shared (linedata, species, temperature, lines, test)   \
  default (none)
  {

    const int num_threads = omp_get_num_threads();
    const int thread_num  = omp_get_thread_num();

    const long start = START + ( thread_num   *ncells_red)/num_threads;
    const long stop  = START + ((thread_num+1)*ncells_red)/num_threads;


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

				test[p] = p;
	  	}
	  }

	} // end of pragma omp parallel

//	MPI_Barrier(MPI_COMM_WORLD);
	
	// Does Allgather have an automatic Barrier?
	
//	long buffer_length = new long[world_size];
//
//	for (int proc = 0; proc < world_size; proc++)
//	{
//		
//	}


	MPI_Allgather (
	  MPI_IN_PLACE,                   // pointer to the data to be send
		0,              // number of elements in the send buffer
		MPI_DATATYPE_NULL,                // type of the send data
		test,                   // pointer to the data to be received
		ncells_red,              // number of elements in the receive buffer
	  MPI_LONG,                // type of the received data
		MPI_COMM_WORLD);
	
//	MPI_Gather (
//	  &(test[START]),                   // pointer to the data to be send
//		ncells_red,              // number of elements in the send buffer
//		MPI_LONG,                // type of the send data
//		test,                   // pointer to the data to be received
//		ncells_red,              // number of elements in the receive buffer
//	  MPI_LONG,                // type of the received data
//		0,                       // root
//		MPI_COMM_WORLD);


//	MPI_Allgather (
//	  &(lines.emissivity[START]),                   // pointer to the data to be send
//		ncells_red,              // number of elements in the send buffer
//		MPI_DOUBLE,                // type of the send data
//		post,                   // pointer to the data to be received
//		ncells_red,              // number of elements in the receive buffer
//	  MPI_DOUBLE,                // type of the received data
//		MPI_COMM_WORLD);

//	MPI_Barrier(MPI_COMM_WORLD);

	if (world_rank == 0)
	{
	  for (long p = 0; p < ncells; p++)
	  {
	  	cout << "test " << p << " " << test[p] << endl;
	  }
	}


	delete [] test;



	return (0);

}
