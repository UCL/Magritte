// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <vector>
using namespace std;

#include "scattering.hpp"


///  Constructor for SCATTERING
///    @param[in] num_of_rays: number of rays
///    @param[in] num_of_freq_scat: number of frequencies in scattering table
/////////////////////////////////////////////////////////////////////////////

SCATTERING :: SCATTERING (long num_of_rays, long num_of_freq_scat)
{

	nrays      = num_of_rays;
	nfreq_scat = num_of_freq_scat;


  // Size and initialize scattering opacity

	opacity_scat.resize (nfreq_scat);

	for (long f = 0; f < nfreq_scat; f++)
	{
		opacity_scat[f] = 0.0;
	}


  // Size and initialize scattering phase function    

	phase_scat.resize (nrays);

	for (long r1 = 0; r1 < nrays; r1++)
	{
	  phase_scat[r1].resize (nrays);

		for (long r2 = 0; r2 < nrays; r2++)
		{
	    phase_scat[r1][r2].resize (nfreq_scat);

			for (long f = 0; f < nfreq_scat; f++)
			{
				phase_scat[r1][r2][f] = 0.0;
			}
		}
	}


}   // END OF CONSTRUCTOR




///  add_opacity: adds the scattering contribution to the opacity
///    @param[in] frequencies: freqiencies at which to evaluate the opacity
///    @param[in/out] chi: opacity to which to add the scattering contribution
//////////////////////////////////////////////////////////////////////////////

int SCATTERING :: add_opacity (vector<double> frequencies, vector<double>& chi)
{
  return (0);  
}
