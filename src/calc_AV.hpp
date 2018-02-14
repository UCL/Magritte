// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __calc_AV_HPP_INCLUDED__
#define __calc_AV_HPP_INCLUDED__

#include "declarations.hpp"


// calc_AV: calculates visual extinction along a ray ray at a grid point
// ---------------------------------------------------------------------

int calc_AV (long ncells, CELL *cell, double *column_tot);

// int calc_AV (long ncells, COLUMN_DENSITIES column, double *AV);

#endif // __calc_AV_HPP_INCLUDED__
