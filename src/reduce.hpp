// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __REDUCE_HPP_INCLUDED__
#define __REDUCE_HPP_INCLUDED__

#include "declarations.hpp"


// reduce: reduce number of cells
// ------------------------------

long reduce (long ncells, CELLS *cells);


// crop: crop spatial range of data
// --------------------------------

int crop (long ncells, CELLS *cells,
          double x_min, double x_max, double y_min, double y_max, double z_min, double z_max);


// density_reduce: reduce number of cell in regions of constant density
// --------------------------------------------------------------------

int density_reduce (long ncells, CELLS *cells, double min_density_change);


// set_ids: determine cell numbers in the reduced grid, return nr of reduced cells
// -------------------------------------------------------------------------------

long set_ids (long ncells, CELLS *cells);


// initialized_reduced_grid: initialize reduced grid
// -------------------------------------------------

int initialize_reduced_grid (long ncells_red, CELLS *cells_red, long ncells, CELLS *cells);


// interpolate: interpolate reduced grid back to original grid
// -----------------------------------------------------------

int interpolate (long ncells_red, CELLS *cells_red, long ncells, CELLS *cells);


#endif // __REDUCE_HPP_INCLUDED__
