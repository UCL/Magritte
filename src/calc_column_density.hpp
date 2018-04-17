// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __CALC_COLUMN_DENSITY_HPP_INCLUDED__
#define __CALC_COLUMN_DENSITY_HPP_INCLUDED__

#include "declarations.hpp"


// calc_column_density: calculate column density for given species for each cell and ray
// -------------------------------------------------------------------------------------

int calc_column_tot (CELLS *cells, RAYS rays);


// calc_column_densities: calculate column densities for species needed in chemistry
//----------------------------------------------------------------------------------

int calc_column_densities (CELLS *cells, RAYS rays, SPECIES species);


// column_density: calculates column density for a species along a ray at a point
// ------------------------------------------------------------------------------

double column_density (CELLS *cells, RAYS rays, long o, int spec, long ray);


#endif // __CALC_COLUMN_DENSITY_HPP_INCLUDED__
