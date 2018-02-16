// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __CELL_LEVEL_POPULATIONS_HPP_INCLUDED__
#define __CELL_LEVEL_POPULATIONS_HPP_INCLUDED__

#include "declarations.hpp"


#if (CELL_BASED)


// level_populations: iteratively calculates level populations
// -----------------------------------------------------------

int cell_level_populations (long ncells, CELL *cell, LINE_SPECIES line_species);


#endif // if CELL_BASED


#endif // __CELL_LEVEL_POPULATIONS_HPP_INCLUDED__
