// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


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
{

	ncells = num_of_cells;


	long index = 0;


  // Count line frequencies  

  for (int l = 0; l < linedata.nlspec; l++)
  {
  	for (int k = 0; k < linedata.nrad[l]; k++)
  	{
  		for (int z = 0; z < N_QUADRATURE_POINTS; z++)
  		{
				index++;
  		}
  	}
  }

	
	/*
	 *  Count other frequencies...
	 */
	

	// Set total number of frequencies

	nfreq     = index;
	nfreq_red = (nfreq + n_vector_lanes - 1) / n_vector_lanes; 


	// Size and initialize all, order and deorder

    	all.resize (ncells);
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
   	all[p].resize (nfreq_red);
  	
  	for (long f = 0; f < nfreq_red; f++)
  	{
      all[p][f] = 0.0;
  	}


	  nr_line[p].resize (linedata.nlspec);

	  for (int l = 0; l < linedata.nlspec; l++)
	  {
	  	nr_line[p][l].resize (linedata.nrad[l]);

	  	for (int k = 0; k < linedata.nrad[l]; k++)
	  	{
	  		nr_line[p][l][k].resize (N_QUADRATURE_POINTS);

	  		for (int z = 0; z < N_QUADRATURE_POINTS; z++)
	  		{
	  			nr_line[p][l][k][z] = 0;
	  		}
	  	}
	  }

  }
	} // end of pragma omp parallel


}   // END OF CONSTRUCTOR




///  reset: specify the frequencies under consideration given the temperature
///    @param[in] linedata: data structure containing the line data
///    @param[in] temperature: data structure containiing the temperature fields
////////////////////////////////////////////////////////////////////////////////

int FREQUENCIES :: reset (const LINEDATA& linedata, const TEMPERATURE& temperature)
{

# pragma omp parallel                         \
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

		Long1   order (nfreq);
    Double1 freqs (nfreq);  

	  for (int l = 0; l < linedata.nlspec; l++)
	  {
			for (int k = 0; k < linedata.nrad[l]; k++)
			{
			  const int i = linedata.irad[l][k];
			  const int j = linedata.jrad[l][k];

			  const double freq_line = linedata.frequency[l](i,j);
        const double width     = profile_width (temperature.gas[p], freq_line);
  	  	
  	    for (long z = 0; z < N_QUADRATURE_POINTS; z++)
        {
  	      freqs[index1] = freq_line + width*H_roots[z];
				  order[index1] = index1;
				
					index1++;
  	    }
  	  }
		}

		
  	/*
  	 *  Set other frequencies...
  	 */


		// Sort frequencies

		heapsort (freqs, order, nfreq);


		long index2 = 0;

	  for (int l = 0; l < linedata.nlspec; l++)
	  {
			for (int k = 0; k < linedata.nrad[l]; k++)
			{
  	    for (long z = 0; z < N_QUADRATURE_POINTS; z++)
        {
					const long    f = index2 / n_vector_lanes;
					const long lane = index2 % n_vector_lanes;

  	      all[p][f].putlane(freqs[index2], lane);

				  nr_line[p][l][k][z] = order[index2];
					index2++;
  	    }
  	  }
		}

	}
	} // end of pragma omp parallel


	return (0);

}
