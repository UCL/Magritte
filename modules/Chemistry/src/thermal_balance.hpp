// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __THERMAL_BALANCE_HPP_INCLUDED__
#define __THERMAL_BALANCE_HPP_INCLUDED__

#include "declarations.hpp"


// thermal_balance: perform thermal balance iterations to determine temperature
// ----------------------------------------------------------------------------

int thermal_balance (CELLS *cells, RAYS rays, SPECIES species, REACTIONS reactions,
                     LINES lines, TIMERS *timers);


// // thermal_balance_Brent: perform thermal balance iterations to determine temperature
// // ----------------------------------------------------------------------------------
//
// int thermal_balance_Brent (long ncells, CELLS *cells, RAYS rays, SPECIES species, REACTIONS reactions,
//                            LINES lines, TIMERS *timers);


#endif // __THERMAL_BALANCE_HPP_INCLUDED__
