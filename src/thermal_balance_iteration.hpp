// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __THERMAL_BALANCE_ITERATION_HPP_INCLUDED__
#define __THERMAL_BALANCE_ITERATION_HPP_INCLUDED__

#include "declarations.hpp"


// thermal_balance_iteration: perform a thermal balance iteration to calculate thermal flux
// ----------------------------------------------------------------------------------------

int thermal_balance_iteration (long ncells, CELL *cell, SPECIES *species, REACTION *reaction, LINE_SPECIES line_species,
                               double *column_H2, double *column_HD, double *column_C, double *column_CO, TIMERS *timers);


#endif // __THERMAL_BALANCE_ITERATION_HPP_INCLUDED__
