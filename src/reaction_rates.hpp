// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __REACTION_RATES_HPP_INCLUDED__
#define __REACTION_RATES_HPP_INCLUDED__

#include "declarations.hpp"


// reaction_rates: Check which kind of reaction and call appropriate rate calculator
// ---------------------------------------------------------------------------------

int reaction_rates (long ncells, CELL *cell, REACTION *reaction, long gridp, double *rad_surface,
                    double *AV, double *column_H2, double *column_HD, double *column_C, double *column_CO);


#endif // __REACTION_RATES_HPP_INCLUDED__
