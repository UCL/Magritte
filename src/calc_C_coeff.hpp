// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __CALC_C_COEFF_HPP_INCLUDED__
#define __CALC_C_COEFF_HPP_INCLUDED__

#include "declarations.hpp"


// calc_C_coeff: calculate collisional coefficients (C_ij) from line data
// ----------------------------------------------------------------------

int calc_C_coeff (long ncells, CELL *cell, double *C_data, double *coltemp, int *icol, int *jcol,
                  double *weight, double *energy, double *C_coeff, long gridp, int lspec );


#endif // __CALC_C_COEFF_HPP_INCLUDED__
