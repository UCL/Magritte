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

#include "types.hpp"
#include "GridTypes.hpp"
#include "radiation.hpp"
#include "frequencies.hpp"
#include "temperature.hpp"
#include "Lines/src/linedata.hpp"


///  LINES: bridge between Levels and RadiativeTransfer calculations
////////////////////////////////////////////////////////////////////

struct LINES
{

	long ncells;          ///< number of cells

	Double3 emissivity;   ///< line emissivity (p,l,k)
	Double3 opacity;      ///< line opacity (p,l,k)


  LINES (const long num_of_cells, const LINEDATA& linedata);   ///< Constructor


  int add_emissivity_and_opacity (FREQUENCIES& frequencies, const TEMPERATURE& temperature,
																 	vDouble1& frequencies_scaled, const long p,
			                            vDouble1& eta, vDouble1& chi) const;

};


#endif // __LINES_HPP_INCLUDED__
