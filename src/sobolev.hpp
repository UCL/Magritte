// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __CELL_SOBOLEV_HPP_INCLUDED__
#define __CELL_SOBOLEV_HPP_INCLUDED__

#include "declarations.hpp"


// sobolev: calculate mean intensity using LVG approximation and escape probabilities
// ----------------------------------------------------------------------------------

int sobolev (long ncells, CELL *cell, LINE_SPECIES line_species,
                  double *Lambda_diagonal, double *mean_intensity_eff,
                  double *source, double *opacity, long o, int lspec, int kr );


#endif // __CELL_SOBOLEV_HPP_INCLUDED__
