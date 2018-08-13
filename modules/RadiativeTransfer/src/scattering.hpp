// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __SCATTERING_HPP_INCLUDED__
#define __SCATTERING_HPP_INCLUDED__


#include <vector>
using namespace std;

#include "types.hpp"
#include "GridTypes.hpp"


struct SCATTERING
{

	const long nrays;        ///< number of rays
	const long nfreq_scat;   ///< number of frequencies in scattering data
	const long nfreq_red;    ///< number of frequencies

	Double1 opacity_scat;    ///< scattering opacity (p,f)


	// Precalculate phase function for all frequencies

	vReal3 phase;      ///< scattering phase function (r1,r2,f)


  SCATTERING (const long num_of_rays,
			        const long num_of_freq_scat,
							const long num_of_freq);       ///< Constructor

  int add_opacity (const vReal& frequencies, vReal& chi) const;

};


#endif // __SCATTERING_HPP_INCLUDED__
