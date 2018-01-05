// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __REDUCE_HPP_INCLUDED__
#define __REDUCE_HPP_INCLUDED__

#include "declarations.hpp"


// reduce: reduce number of cells
// ------------------------------

int reduce (long ncells, CELL *cell, double threshold,
            double x_min, double x_max, double y_min, double y_max, double z_min, double z_max);


// crop: crop spatial range of data
// --------------------------------

int crop (long ncells, CELL *cell,
          double x_min, double x_max, double y_min, double y_max, double z_min, double z_max);


// density_reduce: reduce number of cell in regions of constant density
//---------------------------------------------------------------------

int density_reduce (long ncells, CELL *cell, double threshold);


#endif // __REDUCE_HPP_INCLUDED__
