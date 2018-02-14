// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __CELL_SOBOLEV_HPP_INCLUDED__
#define __CELL_SOBOLEV_HPP_INCLUDED__

#include "declarations.hpp"

#if (CELL_BASED)


// sobolev: calculate mean intensity using LVG approximation and escape probabilities
// ----------------------------------------------------------------------------------

int cell_sobolev (long ncells, CELL *cell, LINE_SPECIES line_species, double *mean_intensity,
                  double *Lambda_diagonal, double *mean_intensity_eff, double *source,
                  double *opacity, long gridp, int lspec, int kr );


#endif // if CELL_BASED

#endif // __CELL_SOBOLEV_HPP_INCLUDED__
