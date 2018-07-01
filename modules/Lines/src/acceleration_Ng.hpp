// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __ACCELERATION_NG_HPP_INCLUDED__
#define __ACCELERATION_NG_HPP_INCLUDED__


#include "levels.hpp"
#include "linedata.hpp"


///  acceleration_Ng: perform a Ng accelerated iteration for level populations
///  All variable names are based on lecture notes by C.P. Dullemond
//////////////////////////////////////////////////////////////////////////////

int acceleration_Ng (LINEDATA& linedata, int l, LEVELS& levels);


///  store_populations: update previous populations
///////////////////////////////////////////////////

int store_populations (LEVELS& levels, int l);


#endif // __ACCELERATION_NG_HPP_INCLUDED__
