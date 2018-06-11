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
	
	vector<vector<vector<vector<long>>>> nr_line;   ///< frequency number corresponing to line (p,l,k,z)


	FREQUENCIES (long num_of_cells, LINEDATA linedata);       ///< Constructor

  int reset (LINEDATA linedata, TEMPERATURE temperature);   ///< Set frequencies

	double integrate_over_line (TEMPERATURE& temperature, const vector<double>& J,
			                        const long p, const int l, const int k);

};


#endif // __FREQUENCIES_HPP_INCLUDED__
