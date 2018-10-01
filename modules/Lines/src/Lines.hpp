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
int Lines (const CELLS<Dimension, Nrays> &cells,
           const LINEDATA                &linedata,
           const SPECIES                 &species,
           const TEMPERATURE             &temperature,
           const FREQUENCIES             &frequencies,
                 LEVELS                  &levels,
                 RADIATION               &radiation);


#include "Lines.tpp"


#endif // __LINES2_HPP_INCLUDED__
