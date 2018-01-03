// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __SOBOLEV_HPP_INCLUDED__
#define __SOBOLEV_HPP_INCLUDED__


#include "../parameters.hpp"
#include "declarations.hpp"

#if (!CELL_BASED)


// sobolev: calculate mean intensity using LVG approximation and escape probabilities
// ----------------------------------------------------------------------------------

int sobolev (long ncells, CELL *cell, EVALPOINT *evalpoint, long *key, long *raytot, long *cum_raytot,
             double *mean_intensity, double *Lambda_diagonal, double *mean_intensity_eff, double *source,
             double *opacity, double *frequency, int *irad, int*jrad, long gridp, int lspec, int kr);


#endif // if not CELL_BASED


#endif // __SOBOLEV_HPP_INCLUDED__
