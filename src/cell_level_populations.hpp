// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __CELL_LEVEL_POPULATIONS_HPP_INCLUDED__
#define __CELL_LEVEL_POPULATIONS_HPP_INCLUDED__


#include "../parameters.hpp"
#include "declarations.hpp"


#if (CELL_BASED)


// level_populations: iteratively calculates level populations
// -----------------------------------------------------------

int cell_level_populations( long ncells, CELL *cell, int *irad, int*jrad, double *frequency,
                            double *A_coeff, double *B_coeff, double *pop,
                            double *C_data, double *coltemp, int *icol, int *jcol,
                            double *weight, double *energy, double *mean_intensity,
                            double *Lambda_diagonal, double *mean_intensity_eff );


#endif // if CELL_BASED


#endif // __CELL_LEVEL_POPULATIONS_HPP_INCLUDED__
