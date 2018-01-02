// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __LEVEL_POPULATIONS_OTF_HPP_INCLUDED__
#define __LEVEL_POPULATIONS_OTF_HPP_INCLUDED__


#include "../parameters.hpp"
#include "declarations.hpp"

#if (!CELL_BASED)


// level_populations: iteratively calculate level populations
// ----------------------------------------------------------

int level_populations_otf( CELL *cell, int *irad, int*jrad, double *frequency,
                           double *A_coeff, double *B_coeff, double *pop,
                           double *C_data, double *coltemp, int *icol, int *jcol,
                           double *temperature_gas, double *temperature_dust,
                           double *weight, double *energy, double *mean_intensity,
                           double *Lambda_diagonal, double *mean_intensity_eff );


#endif // if not CELL_BASED


#endif // __LEVEL_POPULATIONS_OTF_HPP_INCLUDED__
