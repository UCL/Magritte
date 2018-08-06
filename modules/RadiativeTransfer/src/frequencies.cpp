// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <algorithm>
#include <iostream>
#include <vector>
using namespace std;

#include "frequencies.hpp"
#include "GridTypes.hpp"
#include "constants.hpp"
#include "temperature.hpp"
#include "heapsort.hpp"
#include "profile.hpp"
#include "Lines/src/linedata.hpp"


///  Constructor for FREQUENCIES
///    @param[in] num_of_cells: number of cells
///    @param[in] linedata: data structure containing the line data
///////////////////////////////////////////////////////////////////

FREQUENCIES :: FREQUENCIES (const long num_of_cells, const LINEDATA& linedata)
  : ncells    (num_of_cells)
	, nlines    (count_nlines (linedata))
  , nfreq     (count_nfreq (nlines))
	, nfreq_red (count_nfreq_red (nfreq))
{

	// Size and initialize all, order and deorder

   	   nu.resize (ncells);
      dnu.resize (ncells);
	nr_line.resize (ncells);

# pragma omp parallel   \
	shared (linedata)     \
  default (none)
  {

  const int num_threads = omp_get_num_threads();
  const int thread_num  = omp_get_thread_num();

  const long start = (thread_num*ncells)/num_threads;
  const long stop  = ((thread_num+1)*ncells)/num_threads;   // Note brackets


  for (long p = start; p < stop; p++)
  {
   	     nu[p].resize (nfreq_red);
   	    dnu[p].resize (nfreq_red);
	  nr_line[p].resize (linedata.nlspec);

	  for (int l = 0; l < linedata.nlspec; l++)
	  {
	  	nr_line[p][l].resize (linedata.nrad[l]);

	  	for (int k = 0; k < linedata.nrad[l]; k++)
	  	{
	  		nr_line[p][l][k].resize (N_QUADRATURE_POINTS);
	  	}
	  }


    // frequencies.nu has to be initialized (for unused entries)

    for (long f = 0; f < nfreq_red; f++)
    {
       nu[p][f] = 0.0;
      dnu[p][f] = 0.0;
    }

  }
	} // end of pragma omp parallel



  // Find the order of the line center frequencies

  line      .resize (nlines);
  line_index.resize (nlines);

  long index = 0;

	 for (int l = 0; l < linedata.nlspec; l++)
	 {
		for (int k = 0; k < linedata.nrad[l]; k++)
		{
		  line      [index] = linedata.frequency[l][k];
      line_index[index] = index;
      index++;
    }
	}

	// Sort frequencies

  heapsort (line, line_index);


}   // END OF CONSTRUCTOR




#include "folders.hpp"


int FREQUENCIES ::
    write (string tag)
{

		string file_name = output_folder + "frequencies_nu" + tag + ".txt";

    ofstream outputFile (file_name);

    for (long p = 0; p < ncells; p++)
    {
	    for (long f = 0; f < nfreq_red; f++)
      {
#       if (GRID_SIMD)
				  for (int lane = 0; lane < n_simd_lanes; lane++)
				  {
	  		    outputFile << nu[p][f].getlane(lane) << "\t";
				  }
#       else
	  		  outputFile << nu[p][f] << "\t";
#       endif
      }

			outputFile << endl;

	   }

	  outputFile.close ();

  }


///  count_nfreq: count the number of frequencies
///    @param[in] linedata: data structure containing the line data
///////////////////////////////////////////////////////////////////

long FREQUENCIES :: count_nlines (const LINEDATA& linedata)
{

	long index = 0;


  // Count line frequencies

  for (int l = 0; l < linedata.nlspec; l++)
  {
  	for (int k = 0; k < linedata.nrad[l]; k++)
  	{
			index++;
  	}
  }


	return index;

}




///  count_nfreq: count the number of frequencies
///    @param[in] linedata: data structure containing the line data
///////////////////////////////////////////////////////////////////

long FREQUENCIES :: count_nfreq (const long nlines)
{

	long index = 0;


  // Count line frequencies

  index += nlines * N_QUADRATURE_POINTS;

	/*
	 *  Count other frequencies...
	 */


	// Ensure that nfreq is a multiple of n_simd_lanes

  long nfreq_red_tmp = (index + n_simd_lanes - 1) / n_simd_lanes;

	return nfreq_red_tmp * n_simd_lanes;

}




///  count_nfreq_red: count the reduced number of frequencies (= nr of SIMD blocks)
///    @param[in] nfreq: total number of frequency bins
///    @return total number of frequency SIMD blocks
///////////////////////////////////////////////////////////////////////////////////

long FREQUENCIES :: count_nfreq_red (const long nfreq)
{
	return (nfreq + n_simd_lanes - 1) / n_simd_lanes;
}




///  reset: specify the frequencies under consideration given the temperature
///    @param[in] linedata: data structure containing the line data
///    @param[in] temperature: data structure containiing the temperature fields
////////////////////////////////////////////////////////////////////////////////

int FREQUENCIES :: reset (const LINEDATA& linedata, const TEMPERATURE& temperature)
{

# pragma omp parallel                             \
  shared (linedata, temperature, H_roots, cout)   \
	default (none)
  {

  const int num_threads = omp_get_num_threads();
	const int thread_num  = omp_get_thread_num();

  const long start = (thread_num*ncells)/num_threads;
  const long stop  = ((thread_num+1)*ncells)/num_threads;   // Note brackets

	for (long p = start; p < stop; p++)
	{
		long index1 = 0;

		Long1   nmbrs (nfreq);
    Double1 freqs (nfreq);

	  for (int l = 0; l < linedata.nlspec; l++)
	  {
			for (int k = 0; k < linedata.nrad[l]; k++)
			{
			  const double freq_line = linedata.frequency[l][k];
        const double width     = profile_width (temperature.gas[p], freq_line);

  	    for (long z = 0; z < N_QUADRATURE_POINTS; z++)
        {
  	      freqs[index1] = freq_line + width*H_roots[z];
				  nmbrs[index1] = index1;

					index1++;
  	    }
  	  }
		}


  	/*
  	 *  Set other frequencies...
  	 */


		// Sort frequencies

		heapsort (freqs, nmbrs);


    // Set all frequencies nu

	  for (long fl = 0; fl < nfreq; fl++)
	  {
#     if (GRID_SIMD)
		    const long    f = fl / n_simd_lanes;
		    const long lane = fl % n_simd_lanes;
  	    nu[p][f].putlane(freqs[fl], lane);
#		  else
  	    nu[p][f] = freqs[fl];
#     endif
		}


    // Set all frequency increments dnu

#   if (GRID_SIMD)
  	  dnu[p][0].putlane(freqs[1]-freqs[0], 0);
#		else
  	  dnu[p][0] = freqs[1]-freqs[0];
#   endif

	  for (long fl = 1; fl < nfreq-1; fl++)
	  {
#     if (GRID_SIMD)
		    const long    f = fl / n_simd_lanes;
		    const long lane = fl % n_simd_lanes;
  	    dnu[p][f].putlane(0.5*(freqs[fl+1]-freqs[fl-1]), lane);
#		  else
  	    dnu[p][f] = 0.5*(freqs[fl+1]-freqs[fl-1]);
#     endif
		}

#   if (GRID_SIMD)
  	  dnu[p][nfreq_red-1].putlane(freqs[nfreq-1]-freqs[nfreq-2], n_simd_lanes-1);
#		else
  	  dnu[p][nfreq_red-1] = freqs[nfreq-1]-freqs[nfreq-2];
#   endif


		long index2 = 0;

	  for (int l = 0; l < linedata.nlspec; l++)
	  {
			for (int k = 0; k < linedata.nrad[l]; k++)
			{
  	    for (long z = 0; z < N_QUADRATURE_POINTS; z++)
        {
				  nr_line[p][l][k][z] = nmbrs[index2];
					index2++;
  	    }
  	  }
		}

	}
	} // end of pragma omp parallel


	return (0);

}
