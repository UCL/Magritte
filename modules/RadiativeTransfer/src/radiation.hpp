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

	long ncells;                          ///< number of cells
	long nrays;                           ///< number of rays
	long nfreq_red;                       ///< number of frequencies


	vDouble3 u;     ///< u intensity
	vDouble3 v;     ///< v intensity

	vDouble3 U;     ///< U scattered intensity
	vDouble3 V;     ///< V scattered intensity


	RADIATION (const long num_of_cells, const long num_of_rays,
			       const long num_of_freq);                           ///< Constructor


	int resample_U (const FREQUENCIES& frequencies, const long p, const long r,
		              const vDouble1& frequencies_scaled, vDouble1& U_scaled);
		                                

	int resample_V (const FREQUENCIES& frequencies, const long p, const long r,
			            const vDouble1& frequencies_scaled, vDouble1& V_scaled);


};


#endif // __RADIATION_HPP_INCLUDED__
