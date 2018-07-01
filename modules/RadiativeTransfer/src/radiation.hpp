// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________
 

#ifndef __RADIATION_HPP_INCLUDED__
#define __RADIATION_HPP_INCLUDED__


#include <vector>
using namespace std;

#include "frequencies.hpp"
#include "GridTypes.hpp"


///  RADIATION: data structure for the radiation field
//////////////////////////////////////////////////////

struct RADIATION
{

	const long ncells;          ///< number of cells
	const long nrays_red;       ///< reduced number of rays
	const long nfreq_red;       ///< reduced number of frequencies
	const long START_raypair;   ///< reduced number of frequencies


	vReal2 u;   ///< u intensity
	vReal2 v;   ///< v intensity

	vReal2 U;   ///< U scattered intensity
	vReal2 V;   ///< V scattered intensity

	vReal1 test2;   ///< (angular) mean intensity
	vReal1 rec2;   ///< (angular) mean intensity
	vReal1 J;   ///< (angular) mean intensity


	RADIATION (const long num_of_cells, const long num_of_rays,
			       const long num_of_freq,  const long START_raypair_input);   ///< Constructor

	int initialize ();

  long index (const long r, const long p, const long f) const;

  long index (const long p, const long f) const;

	int resample_U (const FREQUENCIES& frequencies, const long p, const long r,
		              const vReal1& frequencies_scaled, vReal1& U_scaled) const;
		                                

	int resample_V (const FREQUENCIES& frequencies, const long p, const long r,
			            const vReal1& frequencies_scaled, vReal1& V_scaled) const;


};


#endif // __RADIATION_HPP_INCLUDED__
