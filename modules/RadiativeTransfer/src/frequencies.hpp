// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __FREQUENCIES_HPP_INCLUDED__
#define __FREQUENCIES_HPP_INCLUDED__

#include <vector>
using namespace std;

#include "types.hpp"
#include "GridTypes.hpp"
#include "temperature.hpp"
#include "Lines/src/linedata.hpp"


struct FREQUENCIES
{

	const long ncells;   ///< number of cells

	const long nfreq;          ///< number of frequencies
	const long nfreq_red;      ///< number of frequencies divided by n_simd_lanes

	vReal2 all;        ///< all considered frequencies at each cell (p,f)
	
	Long4 nr_line;       ///< frequency number corresponing to line (p,l,k,z)


	FREQUENCIES (const long num_of_cells, const LINEDATA& linedata);        ///< Constructor

  int reset (const LINEDATA& linedata, const TEMPERATURE& temperature);   ///< Set frequencies

	static long count_nfreq (const LINEDATA& linedata);

	static long count_nfreq_red (const long nfreq);

};


#endif // __FREQUENCIES_HPP_INCLUDED__
