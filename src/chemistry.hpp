// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __CHEMISTRY_HPP_INCLUDED__
#define __CHEMISTRY_HPP_INCLUDED__


#include "declarations.hpp"


// abundances: calculate abundances for each species at each grid point
// --------------------------------------------------------------------

int chemistry (long ncells, CELL *cell,
               double *temperature_gas, double *temperature_dust, double *rad_surface, double *AV,
               double *column_H2, double *column_HD, double *column_C, double *column_CO);


#endif // __CHEMISTRY_HPP_INCLUDED__
