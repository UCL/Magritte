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

RADIATION :: RADIATION (const long num_of_cells, const long num_of_rays,
		                    const long num_of_freq,  const long START_raypair_input)
	: ncells        (num_of_cells)
  , nrays_red     (num_of_rays)
  , nfreq_red     (num_of_freq)
	, START_raypair (START_raypair_input)
{


	// Size and initialize u, v, U and V

  u.resize (nrays_red);
  v.resize (nrays_red);

  U.resize (nrays_red);
  V.resize (nrays_red);


	for (long r = 0; r < nrays_red; r++)
	{
	  u[r].resize (ncells*nfreq_red);
	  v[r].resize (ncells*nfreq_red);

	  U[r].resize (ncells*nfreq_red);
	  V[r].resize (ncells*nfreq_red);
	}

	J.resize (ncells*nfreq_red);
	test2.resize (2000);
	rec2.resize (3000);


	initialize();



}   // END OF CONSTRUCTOR



long RADIATION ::
     index (const long r, const long p, const long f) const
{
	return f + (p + (r-START_raypair)*ncells)*nfreq_red;
}

long RADIATION ::
     index (const long p, const long f) const
{
	return f + p*nfreq_red;
}

int RADIATION ::
    initialize ()
{

	for (long r = 0; r < nrays_red; r++)
	{
	
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
	    for (long f = 0; f < nfreq_red; f++)
      {
				u[r][index(p,f)] = 0.0;
        v[r][index(p,f)] = 0.0;
				
				U[r][index(p,f)] = 0.0;
        V[r][index(p,f)] = 0.0;
      }
	  }
	  } // end of pragma omp parallel
	}


#   pragma omp parallel   \
    default (none)
    {

    const int num_threads = omp_get_num_threads();
    const int thread_num  = omp_get_thread_num();

    const long start = (thread_num*ncells)/num_threads;
    const long stop  = ((thread_num+1)*ncells)/num_threads;   // Note brackets


    for (long p = start; p < stop; p++)
    {
	    for (long f = 0; f < nfreq_red; f++)
      {
        J[index(p,f)] = 0.0;
      }
	  }
	  } // end of pragma omp parallel


	return (0);

}



int RADIATION :: resample_U (const FREQUENCIES& frequencies, const long p, const long r,
	                           const vReal1& frequencies_scaled, vReal1& U_scaled) const
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
		                         const vReal1& frequencies_scaled, vReal1& V_scaled) const
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
