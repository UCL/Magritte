// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __SOBOLEV_HPP_INCLUDED__
#define __SOBOLEV_HPP_INCLUDED__

#include "declarations.hpp"


// sobolev: calculate mean intensity using LVG approximation and escape probabilities
// ----------------------------------------------------------------------------------

int sobolev (long ncells, CELLS *cells, RAYS rays, LINES lines,
             double *Lambda_diagonal, double *mean_intensity_eff,
             double *source, double *opacity, long o, int ls, int kr );


#endif // __SOBOLEV_HPP_INCLUDED__
