// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __BOUND_HPP_INCLUDED__
#define __BOUND_HPP_INCLUDED__

#include "declarations.hpp"


// bound_cube: put cube boundary around  grid
// ------------------------------------------

long bound_cube (long ncells, CELL *cell_init, CELL *cell_full, long size);


// bound_sphere: put sphere boundary around  grid
// -----------------------------------------------

long bound_sphere (long ncells, CELL *cell_init, CELL *cell_full, long nboundary_cells)


#endif // __BOUND_HPP_INCLUDED__
