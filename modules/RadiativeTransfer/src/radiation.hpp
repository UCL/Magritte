// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________
 

#ifndef __RADIATION_HPP_INCLUDED__
#define __RADIATION_HPP_INCLUDED__


#include <vector>
using namespace std;

#include "frequencies.hpp"


///  RADIATION: data structure for the radiation field
//////////////////////////////////////////////////////

struct RADIATION
{

	long ncells;                          ///< number of cells
	long nrays;                           ///< number of rays
	long nfreq;                           ///< number of frequencies


	vector<vector<vector<double>>> u;     ///< u intensity
	vector<vector<vector<double>>> v;     ///< v intensity

	vector<vector<vector<double>>> U;     ///< U scattered intensity
	vector<vector<vector<double>>> V;     ///< V scattered intensity


	RADIATION (const long num_of_cells, const long num_of_rays,
			       const long num_of_freq);                           ///< Constructor


	int resample_U (const FREQUENCIES& frequencies, const long p, const long r,
		              const vector<double>& frequencies_scaled,
		  						vector<double>& U_scaled);   ///< U frequency interpolator
		                                

	int resample_V (const FREQUENCIES& frequencies, const long p, const long r,
			            const vector<double>& frequencies_scaled,
							 	  vector<double>& V_scaled);   ///< V frequency interpolator


};


#endif // __RADIATION_HPP_INCLUDED__
