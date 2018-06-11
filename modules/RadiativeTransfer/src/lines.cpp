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
#include "constants.hpp"
#include "temperature.hpp"
#include "frequencies.hpp"
#include "radiation.hpp"
#include "profile.hpp"
#include "Lines/src/linedata.hpp"
#include "Lines/src/levels.hpp"


///  Constructor for LINES
//////////////////////////

LINES :: LINES (long num_of_cells, LINEDATA& linedata)
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




///  get_emissivity_and_opacity
///    @param[in] linedata: data structure containing the line data
///    @param[in] levels: data structure containing the level populations
/////////////////////////////////////////////////////////////////////////

int LINES :: get_emissivity_and_opacity (LINEDATA& linedata, LEVELS& levels)
{

	for (int l = 0; l < linedata.nlspec; l++)
	{
    for (int k = 0; k < linedata.nrad[l]; k++)
	  {
	    int i = linedata.irad[l][k];
		  int j = linedata.jrad[l][k];

      double hv_4pi = HH * linedata.frequency[l](i,j) / (4.0*PI);

#     pragma omp parallel                             \
      shared (linedata, levels, l, k, i, j, hv_4pi, cout)   \
	  	default (none)
	    {

	    const int num_threads = omp_get_num_threads();
			const int thread_num  = omp_get_thread_num();

  	  const long start = (thread_num*ncells)/num_threads;
		  const long stop  = ((thread_num+1)*ncells)/num_threads;   // Note brackets


		  for (long p = start; p < stop; p++)
		  {
		    emissivity[p][l][k] = hv_4pi * linedata.A[l](i,j) * levels.population[p][l](i);

		       opacity[p][l][k] = hv_4pi * ( levels.population[p][l](j) * linedata.B[l](j,i)
  					                             - levels.population[p][l](i) * linedata.B[l](i,j) );

    //    cout << levels.population[p][l](i) << endl;
    //    cout << levels.population[p][l](j) << endl;

		//		cout << "emissivity " << p << " " << l << " " << k << " " << emissivity[p][l][k] << endl;
		//		cout << "   opacity " << p << " " << l << " " << k << " " <<    opacity[p][l][k] << endl;
		//		cout << opacity[p][l][k] << endl;

		  } 
  	  } // end of OpenMP parallel region
    }
  }


  return (0);

}




///  add_emissivity_and_opacity
///////////////////////////////

int LINES :: add_emissivity_and_opacity (FREQUENCIES& frequencies, TEMPERATURE& temperature, 
		                                     vector<double>& frequencies_scaled, long p,
		                                     vector<double>& eta, vector<double>& chi)
{

  // For all frequencies
	
	for (long f = 0; f < frequencies.nfreq; f++)
	{


	  // For all lines

    for (int l = 0; l < frequencies.nr_line[p].size(); l++)
	  {
	  	for (int k = 0; k < frequencies.nr_line[p][l].size(); k++)
	  	{
        const double lower = frequencies.nr_line[p][l][k][0];   // lowest frequency for the line
        const double upper = frequencies.nr_line[p][l][k][3];   // highest frequency for the line

				if (    (frequencies.all[p][lower] <  frequencies_scaled[f])
					   && (frequencies.all[p][upper] >  frequencies_scaled[f]) )
				{
          const double freq_line = 0.5 * (   frequencies.all[p][frequencies.nr_line[p][l][k][1]]
	  		  		                             + frequencies.all[p][frequencies.nr_line[p][l][k][2]] );

	  		  const double line_profile = profile (temperature.gas[p], freq_line, frequencies_scaled[f]);

	  	    eta[f] += emissivity[p][l][k] * line_profile;
	  	    chi[f] +=    opacity[p][l][k] * line_profile;
			  }
	  	}
	  }


	}	


	return (0);

}
