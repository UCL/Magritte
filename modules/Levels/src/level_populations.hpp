// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __LEVEL_POPULATIONS_HPP_INCLUDED__
#define __LEVEL_POPULATIONS_HPP_INCLUDED__

#include "declarations.hpp"


// level_populations: iteratively calculates level populations
// -----------------------------------------------------------

int level_populations (long ncells, CELLS *cells, RAYS rays,
                       SPECIES species, LINES lines);


#endif // __LEVEL_POPULATIONS_HPP_INCLUDED__
