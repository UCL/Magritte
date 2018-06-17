// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <omp.h>
#include <vector>
#include <iostream>
using namespace std;

#include "radiation.hpp"
#include "GridTypes.hpp"
#include "frequencies.hpp"
#include "interpolation.hpp"


///  Constructor for RADIATION
//////////////////////////////

RADIATION :: RADIATION (const long num_of_cells, const long num_of_rays, const long num_of_freq)
{

	ncells     = num_of_cells;
	nrays      = num_of_rays;
	nfreq_red  = num_of_freq;

	
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
	    u[r][p].resize (nfreq_red);
	    v[r][p].resize (nfreq_red);

	    U[r][p].resize (nfreq_red);
	    V[r][p].resize (nfreq_red);

	    for (long f = 0; f < nfreq_red; f++)
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




int RADIATION :: resample_U (const FREQUENCIES& frequencies, const long p, const long r,
	                           const vDouble1& frequencies_scaled, vDouble1& U_scaled)
{
  long start = 0;
	long stop  = nfreq_red;

//	for (long f = 0; f < nfreq; f++)
//	{
//		cout << U[r][p][f] << endl;
//		U_scaled[f] = 0.0;
//	}

	//resample (frequencies.all[p], U[r][p], start, stop, frequencies_scaled, U_scaled);
		
 	return (0);
}




int RADIATION :: resample_V (const FREQUENCIES& frequencies, const long p, const long r,
		                         const vDouble1& frequencies_scaled, vDouble1& V_scaled)
{
  long start = 0;
	long stop  = nfreq_red;

//	for (long f = 0; f < nfreq; f++)
//	{
//		V_scaled[f] = 0.0;
//	}

	//resample (frequencies.all[p], V[r][p], start, stop, frequencies_scaled, V_scaled);
		
 	return (0);
}
