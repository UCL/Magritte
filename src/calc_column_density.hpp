// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __CALC_COLUMN_DENSITY_HPP_INCLUDED__
#define __CALC_COLUMN_DENSITY_HPP_INCLUDED__

#include "declarations.hpp"


// calc_column_density: calculate column density for given species for each cell and ray
// -------------------------------------------------------------------------------------

int calc_column_density (long ncells, CELL *cell, double *column_density, int spec);


// calc_column_densities: calculate column densities for species needed in chemistry
//----------------------------------------------------------------------------------

int calc_column_densities (long ncells, CELL *cell, double *column_H2, double *column_HD,
                           double *column_C, double *column_CO);


#if (!CELL_BASED)


  // column_density: calculate column density for one species along one ray
  // ----------------------------------------------------------------------

  double column_density (long ncells, CELL *cell, EVALPOINT *evalpoint, long *key, long *raytot,
                         long *cum_raytot, long gridp, int spec, long ray);


#else

  // cell_column_density: calculates column density for a species along a ray at a point
  // --------------------------------------------------------------------------------------------

  double cell_column_density (long ncells, CELL *cell, long gridp, int spec, long ray);


#endif


#endif // __CALC_COLUMN_DENSITY_HPP_INCLUDED__
