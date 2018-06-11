// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __LINES_HPP_INCLUDED__
#define __LINES_HPP_INCLUDED__


#include <vector>
using namespace std;
#include <Eigen/Core>
using namespace Eigen;

#include "radiation.hpp"
#include "frequencies.hpp"
#include "temperature.hpp"
#include "Lines/src/linedata.hpp"
#include "Lines/src/levels.hpp"


///  LINES: bridge between Levels and RadiativeTransfer calculations
////////////////////////////////////////////////////////////////////

struct LINES
{

	long ncells;                                 ///< number of cells

	vector<vector<vector<double>>> emissivity;   ///< line emissivity (p,l,k)
	vector<vector<vector<double>>> opacity;      ///< line opacity (p,l,k)


  LINES (long num_of_cells, LINEDATA& linedata);   ///< Constructor


	int get_emissivity_and_opacity (LINEDATA& linedata, LEVELS& levels);


  int add_emissivity_and_opacity (FREQUENCIES& frequencies, TEMPERATURE& temperature,
																 	vector<double>& frequencies_scaled, long p,
			                            vector<double>& eta, vector<double>& chi);

};


#endif // __LINES_HPP_INCLUDED__
