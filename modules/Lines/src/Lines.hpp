// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __LINES2_HPP_INCLUDED__
#define __LINES2_HPP_INCLUDED__


#include <vector>
using namespace std;
#include <Eigen/Core>
using namespace std;

#include "Lines.hpp"
#include "levels.hpp"
#include "linedata.hpp"
#include "acceleration_Ng.hpp"
#include "RadiativeTransfer/src/species.hpp"
#include "RadiativeTransfer/src/temperature.hpp"
#include "RadiativeTransfer/src/RadiativeTransfer.hpp"
#include "RadiativeTransfer/src/cells.hpp"
#include "RadiativeTransfer/src/lines.hpp"
#include "RadiativeTransfer/src/radiation.hpp"
#include "RadiativeTransfer/src/frequencies.hpp"


///  Lines: iteratively calculates level populations
////////////////////////////////////////////////////

template <int Dimension, long Nrays>
int Lines (CELLS<Dimension, Nrays>& cells, LINEDATA& linedata, SPECIES& species,
		       TEMPERATURE& temperature, FREQUENCIES& frequencies, LEVELS& levels,
					 RADIATION& radiation);


int calc_level_populations (LINEDATA& linedata, LINES& lines, LEVELS& levels, SPECIES& species,
		                        FREQUENCIES& frequencies, TEMPERATURE& temperature, vector<vector<double>>& J,
														vector<bool>& not_converged, vector<long>& n_not_converged,
														bool some_not_converged, long Nrays);

#include "Lines.tpp"


#endif // __LINES2_HPP_INCLUDED__
