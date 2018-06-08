// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <omp.h>
#include <vector>
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
      shared (linedata, levels, l, k, i, j, hv_4pi)   \
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

  for (int l = 0; l < frequencies.nr_line.size(); l++)
	{
		for (int k = 0; k < frequencies.nr_line[l].size(); k++)
		{
      const double lower = frequencies.order[p][frequencies.nr_line[l][k][0]];
      const double upper = frequencies.order[p][frequencies.nr_line[l][k][3]];

      const double freq_line = 0.5 * (   frequencies.all[p][frequencies.nr_line[l][k][1]]
					                             + frequencies.all[p][frequencies.nr_line[l][k][2]] );

			for (long index = lower; index <= upper; index++)
			{
				const double line_profile = profile (temperature.gas[p], freq_line, frequencies_scaled[frequencies.deorder[p][index]]);

		    eta[frequencies.deorder[p][index]] += emissivity[p][l][k] * line_profile;
		    chi[frequencies.deorder[p][index]] +=    opacity[p][l][k] * line_profile;
			}
		}
	}		


	return (0);

}




///  J_eff: effective mean intensity in a line
//////////////////////////////////////////////

double LINES :: J_eff (FREQUENCIES& frequencies, TEMPERATURE& temperature,
		                   vector<vector<double>>& J, long p, int l, int k)
{

  const vector<long> freq_nrs = frequencies.nr_line[l][k];

  const double freq_line = 0.5 * (   frequencies.all[p][freq_nrs[1]]
			                             + frequencies.all[p][freq_nrs[2]] );


	double result = 0.0;

	for (long z = 0; z < 4; z++)
	{
    result += H_4_weights[z] / profile_width (temperature.gas[p], freq_line) * J[p][freq_nrs[z]];
	}


  return result;

}
