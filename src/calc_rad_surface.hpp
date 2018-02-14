// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __calc_rad_surface_HPP_INCLUDED__
#define __calc_rad_surface_HPP_INCLUDED__


// calc_rad_surface: calculates UV radiation surface for each ray at each grid point
// ---------------------------------------------------------------------------------

int calc_rad_surface (long ncells, CELL *cell, double *G_external);


#endif // __calc_rad_surface_HPP_INCLUDED__
