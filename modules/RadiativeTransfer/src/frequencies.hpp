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

	long ncells;      ///< number of cells
	long nfreq;       ///< number of frequencies
	long nfreq_red;   ///< number of frequencies divided by n_vector_lanes

	vDouble2 all;     ///< all considered frequencies at each cell (p,f)
	
	Long4 nr_line;    ///< frequency number corresponing to line (p,l,k,z)


	FREQUENCIES (const long num_of_cells, const LINEDATA& linedata);        ///< Constructor

  int reset (const LINEDATA& linedata, const TEMPERATURE& temperature);   ///< Set frequencies

};


#endif // __FREQUENCIES_HPP_INCLUDED__
