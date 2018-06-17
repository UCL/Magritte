// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


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
{

  ncells = num_of_cells;


  // Size and initialize emissivity, opacity and freq

 	emissivity.resize (ncells);
 	   opacity.resize (ncells);

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




///  add_emissivity_and_opacity
///////////////////////////////

int LINES ::
    add_emissivity_and_opacity (FREQUENCIES& frequencies, const TEMPERATURE& temperature, 
		                            vDouble1& frequencies_scaled, const long p,
		                            vDouble1& eta, vDouble1& chi) const
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

		    const long    f_lower = lower / n_vector_lanes;
		    const long lane_lower = lower % n_vector_lanes;

		    const long    f_upper = upper / n_vector_lanes;
		    const long lane_upper = upper % n_vector_lanes;
		

				if (   !(frequencies.all[p][f_lower].getlane(lane_lower) > frequencies_scaled[f].getlane(n_vector_lanes-1))
					  || !(frequencies.all[p][f_upper].getlane(lane_upper) < frequencies_scaled[f].getlane(0)) )
				{
		      const long    f_line = freq_nrs[NR_LINE_CENTER] / n_vector_lanes;
		      const long lane_line = freq_nrs[NR_LINE_CENTER] % n_vector_lanes;

          const double freq_line =  frequencies.all[p][f_line].getlane(lane_line);

	  		  const vDouble line_profile = profile (temperature.gas[p], freq_line, frequencies_scaled[f]);

	  	    eta[f] += emissivity[p][l][k] * line_profile;
	  	    chi[f] +=    opacity[p][l][k] * line_profile;
			  }
	  	}
	  }


	}	


	return (0);

}
