// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <omp.h>
#include <vector>
using namespace std;

#include "radiation.hpp"
#include "frequencies.hpp"
#include "interpolation.hpp"


///  Constructor for RADIATION
//////////////////////////////

RADIATION :: RADIATION (long num_of_cells, long num_of_rays, long num_of_freq)
{

	ncells = num_of_cells;
	nrays  = num_of_rays;
	nfreq  = num_of_freq;

	
	// Size and initialize u, v, U and V

  u.resize (nrays);
	v.resize (nrays);
	
	U.resize (nrays);
	V.resize (nrays);

	for (long r = 0; r < nrays; r++)
	{
	  u[r].resize (ncells);
	  v[r].resize (ncells);

	  U[r].resize (ncells);
	  V[r].resize (ncells);
		
#   pragma omp parallel   \
		shared (r)            \
    default (none)
    {

    const int num_threads = omp_get_num_threads();
    const int thread_num  = omp_get_thread_num();

    const long start = (thread_num*ncells)/num_threads;
    const long stop  = ((thread_num+1)*ncells)/num_threads;   // Note brackets


    for (long p = start; p < stop; p++)
    {
	    u[r][p].resize (nfreq);
	    v[r][p].resize (nfreq);

	    U[r][p].resize (nfreq);
	    V[r][p].resize (nfreq);

	    for (long f = 0; f < nfreq; f++)
      {
        u[r][p][f] = 0.0;
        v[r][p][f] = 0.0;

        U[r][p][f] = 0.0;
        V[r][p][f] = 0.0;
      }	
	  }
	  } // end of pragma omp parallel
	}


}   // END OF CONSTRUCTOR




int RADIATION :: resample_U (FREQUENCIES frequencies, long p, long r,
	                           vector<double> frequencies_scaled,
		                         vector<double>& U_scaled)
{
  long start = 0;
	long stop  = nfreq;

	resample (frequencies.all[p], U[r][p], start, stop, frequencies_scaled, U_scaled);
		
 	return (0);
}




int RADIATION :: resample_V (FREQUENCIES frequencies, long p, long r,
		                         vector<double> frequencies_scaled,
		                         vector<double>& V_scaled)
{
  long start = 0;
	long stop  = nfreq;

	resample (frequencies.all[p], V[r][p], start, stop, frequencies_scaled, V_scaled);
		
 	return (0);
}
