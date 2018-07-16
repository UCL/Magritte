// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <mpi.h>
#include <omp.h>
#include <vector>
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

LINES :: LINES (const long num_of_cells, const LINEDATA& linedata)
	: ncells   (num_of_cells)
	, nlspec   (linedata.nlspec)
	, nrad     (linedata.nrad)
	, nrad_cum (get_nrad_cum (nrad))
	, nrad_tot (get_nrad_tot (nrad))
{



  // Size and initialize emissivity, opacity and freq

 	emissivity.resize (ncells);
 	   opacity.resize (ncells);

 	emissivity_vec.resize (ncells*nrad_tot);
 	   opacity_vec.resize (ncells*nrad_tot);

# pragma omp parallel \
  shared (linedata)   \
	default (none)
  {

  const int num_threads = omp_get_num_threads();
	const int thread_num  = omp_get_thread_num();

  const long start = (thread_num*ncells)/num_threads;
  const long stop  = ((thread_num+1)*ncells)/num_threads;   // Note brackets

	for (long p = start; p < stop; p++)
	{
		emissivity[p].resize (linedata.nlspec);
		   opacity[p].resize (linedata.nlspec);

	  for (int l = 0; l < linedata.nlspec; l++)
	  {
		  emissivity[p][l].resize (linedata.nrad[l]);
		     opacity[p][l].resize (linedata.nrad[l]);

		  for (int k = 0; k < linedata.nrad[l]; k++)
		  {
			  emissivity[p][l][k] = 0.0;
			     opacity[p][l][k] = 0.0;
  	  }
		}
	
	}
	} // end of pragma omp parallel


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


long LINES ::
     index (const long p, const int l, const int k) const
{
	return k + nrad_cum[l] + p*nrad_tot;
}



///  add_emissivity_and_opacity
///////////////////////////////

int LINES ::
    add_emissivity_and_opacity (FREQUENCIES& frequencies, const TEMPERATURE& temperature, 
		                            vReal1& frequencies_scaled, const long p,
		                            vReal1& eta, vReal1& chi) const
{

	// !!! RETHINK THIS !!!!


  // For all frequencies
	
	for (long f = 0; f < frequencies.nfreq_red; f++)
	{


	  // For all lines

    for (int l = 0; l < frequencies.nr_line[p].size(); l++)
	  {
	  	for (int k = 0; k < frequencies.nr_line[p][l].size(); k++)
	  	{
				const Long1 freq_nrs = frequencies.nr_line[p][l][k];

        const long lower = freq_nrs[0];                       // lowest frequency in line
        const long upper = freq_nrs[N_QUADRATURE_POINTS-1];   // highest frequency in line
#       if (GRID_SIMD)
		    const long    f_lower = lower / n_simd_lanes;
		    const long lane_lower = lower % n_simd_lanes;

		    const long    f_upper = upper / n_simd_lanes;
		    const long lane_upper = upper % n_simd_lanes;
		

				if (   !(frequencies.all[p][f_lower].getlane(lane_lower) > frequencies_scaled[f].getlane(n_simd_lanes-1))
					  || !(frequencies.all[p][f_upper].getlane(lane_upper) < frequencies_scaled[f].getlane(0)) )
#       else
				if (   !(frequencies.all[p][lower] > frequencies_scaled[f])
					  || !(frequencies.all[p][upper] < frequencies_scaled[f]) )
#       endif
				{
#         if (GRID_SIMD)
		      const long    f_line = freq_nrs[NR_LINE_CENTER] / n_simd_lanes;
		      const long lane_line = freq_nrs[NR_LINE_CENTER] % n_simd_lanes;

          const double freq_line = frequencies.all[p][f_line].getlane(lane_line);
#         else
          const double freq_line = frequencies.all[p][freq_nrs[NR_LINE_CENTER]];
# 				endif

	  		  const vReal line_profile = profile (temperature.gas[p], freq_line, frequencies_scaled[f]);

	  	    eta[f] += emissivity[p][l][k] * line_profile;
	  	    chi[f] +=    opacity[p][l][k] * line_profile;

	  	    eta[f] += emissivity_vec[index(p,l,k)] * line_profile;
	  	    chi[f] +=    opacity_vec[index(p,l,k)] * line_profile;
			  }
	  	}
	  }


	}	


  // For all frequencies
	
	for (long f = 0; f < frequencies.nfreq_red; f++)
	{
#   if (GRID_SIMD)
			for (int lane = 0; lane < n_simd_lanes; lane++)
			{
		    if (chi[f].getlane(lane) == 0.0)
				{
          chi[f].getlane(lane) = 1.0E-30;
				}
			}
#   else
		  if (chi[f] == 0.0)
			{
        chi[f] = 1.0E-30;
			}
#   endif
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
		              emissivity_vec.data(),   // pointer to the data to be received
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
              		opacity_vec.data(),      // pointer to the data to be received
              		buffer_lengths,          // number of elements in receive buffer
                  displacements,           // displacements between data blocks 
              	  MPI_DOUBLE,              // type of the received data
              		MPI_COMM_WORLD);

	assert (ierr_op == 0);


	delete [] buffer_lengths;
	delete [] displacements;
  

	return (0);

}
