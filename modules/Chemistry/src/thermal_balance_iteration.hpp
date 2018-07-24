// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __THERMAL_BALANCE_ITERATION_HPP_INCLUDED__
#define __THERMAL_BALANCE_ITERATION_HPP_INCLUDED__

#include "declarations.hpp"


// thermal_balance_iteration: perform a thermal balance iteration to calculate thermal flux
// ----------------------------------------------------------------------------------------

int thermal_balance_iteration (CELLS *cells, RAYS rays, SPECIES species, REACTIONS reactions,
                               LINES lines, TIMERS *timers);


#endif // __THERMAL_BALANCE_ITERATION_HPP_INCLUDED__