// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <mpi.h>
#include <omp.h>
#include <assert.h>
#include <fstream>
#include <iostream>
using namespace std;
#include <Eigen/Core>
using namespace Eigen;

#include "lines.hpp"
#include "types.hpp"
#include "GridTypes.hpp"
#include "constants.hpp"
#include "temperature.hpp"
#include "frequencies.hpp"
#include "radiation.hpp"
#include "profile.hpp"
#include "Lines/src/linedata.hpp"


///  Constructor for LINES
//////////////////////////

LINES ::
LINES (const long num_of_cells, const LINEDATA& linedata)
	: ncells   (num_of_cells)
	, nlspec   (linedata.nlspec)
	, nrad     (linedata.nrad)
	, nrad_cum (get_nrad_cum (nrad))
	, nrad_tot (get_nrad_tot (nrad))
{

 	emissivity.resize (ncells*nrad_tot);
 	   opacity.resize (ncells*nrad_tot);


}   // END OF CONSTRUCTOR


Int1 LINES ::
		 get_nrad_cum (const Int1 nrad)
{

	Int1 result (nrad.size(), 0);

	for (int l = 1; l < nrad.size(); l++)
	{
	  result[l] = result[l-1] + nrad[l-1];
	}

	return result;

}

int LINES ::
		get_nrad_tot (const Int1 nrad)
{

	int result = 0;

	for (int l = 0; l < nrad.size(); l++)
	{
	  result += nrad[l];
	}

	return result;

}





int LINES ::
    print (string output_folder,
           string tag           ) const
{

  int world_rank;
  MPI_Comm_rank (MPI_COMM_WORLD, &world_rank);


  if (world_rank == 0)
  {
    for (int l = 0; l < nlspec; l++)
    {
      string eta_file = output_folder + "eta_" + to_string (l) + tag + ".txt";
      string chi_file = output_folder + "chi_" + to_string (l) + tag + ".txt";

      ofstream eta_outputFile (eta_file);
      ofstream chi_outputFile (chi_file);

      for (long p = 0; p < ncells; p++)
      {
        for (int k = 0; k < nrad[l]; k++)
        {
          const long ind = index(p,l,k);

          eta_outputFile << emissivity[ind] << "\t";
          chi_outputFile <<    opacity[ind] << "\t";
        }

        eta_outputFile << endl;
        chi_outputFile << endl;
      }

      eta_outputFile.close ();
      chi_outputFile.close ();

      cout << "Written files:" << endl;
      cout << eta_file         << endl;
      cout << chi_file         << endl;
    }
  }


  return (0);

}






int LINES ::
    mpi_allgatherv ()
{

	// Get number of processes

  int world_size;
	MPI_Comm_size (MPI_COMM_WORLD, &world_size);


	// Extract the buffer lengths and displacements

	int *buffer_lengths = new int[world_size];
	int *displacements  = new int[world_size];


	for (int w = 0; w < world_size; w++)
	{
	  long START_w = ( w   *ncells)/world_size;
	  long STOP_w  = ((w+1)*ncells)/world_size;

		long ncells_red_w = STOP_w - START_w;

		buffer_lengths[w] = ncells_red_w * nrad_tot;
	}

	displacements[0] = 0;

	for (int w = 1; w < world_size; w++)
	{
		displacements[w] = buffer_lengths[w-1];
	}


	// Call MPI to gather the emissivity data

  int ierr_em =	MPI_Allgatherv (
	                MPI_IN_PLACE,            // pointer to data to be send (here in place)
		              0,                       // number of elements in the send buffer
		              MPI_DATATYPE_NULL,       // type of the send data
		              emissivity.data(),       // pointer to the data to be received
		              buffer_lengths,          // number of elements in receive buffer
                  displacements,           // displacements between data blocks
	                MPI_DOUBLE,              // type of the received data
		              MPI_COMM_WORLD);

	assert (ierr_em == 0);


	// Call MPI to gather the opacity data

	int ierr_op = MPI_Allgatherv (
              	  MPI_IN_PLACE,            // pointer to data to be send (here in place)
              		0,                       // number of elements in the send buffer
              		MPI_DATATYPE_NULL,       // type of the send data
              		opacity.data(),          // pointer to the data to be received
              		buffer_lengths,          // number of elements in receive buffer
                  displacements,           // displacements between data blocks
              	  MPI_DOUBLE,              // type of the received data
              		MPI_COMM_WORLD);

	assert (ierr_op == 0);


	delete [] buffer_lengths;
	delete [] displacements;


	return (0);

}
