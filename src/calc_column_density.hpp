// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __CALC_COLUMN_DENSITY_HPP_INCLUDED__
#define __CALC_COLUMN_DENSITY_HPP_INCLUDED__

#include "../parameters.hpp"
#include "Magritte_config.hpp"
#include "declarations.hpp"


// calc_column_density: calculate column density for given species for each cell and ray
// -------------------------------------------------------------------------------------

int calc_column_density (long ncells, CELL *cell, double *column_density, int spec);


// calc_column_densities: calculate column densities for species needed in chemistry
//----------------------------------------------------------------------------------

int calc_column_densities( CELL *cell, double *column_H2, double *column_HD,
                           double *column_C, double *column_CO );


// column_density: calculate column density for one species along one ray
// ----------------------------------------------------------------------

double column_density_at_point( CELL *cell, EVALPOINT *evalpoint, long *key,
                                long *raytot, long *cum_raytot, long gridp, int spec, long ray );


#endif // __CALC_COLUMN_DENSITY_HPP_INCLUDED__
