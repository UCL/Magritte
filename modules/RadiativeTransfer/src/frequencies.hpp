// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __FREQUENCIES_HPP_INCLUDED__
#define __FREQUENCIES_HPP_INCLUDED__

#include <vector>
using namespace std;

#include "temperature.hpp"
#include "Lines/src/linedata.hpp"


struct FREQUENCIES
{

	long ncells;                            ///< number of cells
	long nfreq;                             ///< number of frequencies

	vector<vector<double>> all;             ///< all considered frequencies at each cell (p,f)

	vector<vector<long>> order;             ///< frequency numbers in ascending order (p,f)  
	vector<vector<long>> deorder;           ///< frequency numbers ordered as in memory (p,f) 
	
	vector<vector<vector<long>>> nr_line;   ///< frequency number corresponing to line (l,k,z)


	FREQUENCIES (long num_of_cells, LINEDATA linedata);       ///< Constructor

  int reset (LINEDATA linedata, TEMPERATURE temperature);   ///< Set frequencies

};


#endif // __FREQUENCIES_HPP_INCLUDED__
