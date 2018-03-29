// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __BOUND_HPP_INCLUDED__
#define __BOUND_HPP_INCLUDED__

#include "declarations.hpp"


// bound_cube: put cube boundary around  grid
// ------------------------------------------

long bound_cube (CELLS *cells_init, CELLS *cells_full,
                 long size_x, long size_y, long size_z);


// bound_sphere: put sphere boundary around  grid
// -----------------------------------------------

long bound_sphere (CELLS *cells_init, CELLS *cells_full,
                   long nboundary_cells);


#endif // __BOUND_HPP_INCLUDED__
