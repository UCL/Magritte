// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __THERMAL_BALANCE_HPP_INCLUDED__
#define __THERMAL_BALANCE_HPP_INCLUDED__

#include "declarations.hpp"


// preliminary_chemistry: prform preliminary chemistry  iterations
// ---------------------------------------------------------------

int preliminary_chemistry (long ncells, CELL *cell, SPECIES *species, REACTION *reaction, TIMERS *timers);


// thermal_balance: perform thermal balance iterations to determine temperature
// ----------------------------------------------------------------------------

int thermal_balance_std (long ncells, CELL *cell, SPECIES *species, REACTION *reaction,
                         LINE_SPECIES line_species, TIMERS *timers, const double precision);


// thermal_balance_Brent: perform thermal balance iterations to determine temperature
// ----------------------------------------------------------------------------------

int thermal_balance_Brent (long ncells, CELL *cell, SPECIES *species, REACTION *reaction,
                           LINE_SPECIES line_species, TIMERS *timers, const double precision);


#endif // __THERMAL_BALANCE_HPP_INCLUDED__
