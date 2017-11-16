/* Frederik De Ceuster - University College London & KU Leuven                                   */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/* Header for calc_column_density.cpp                                                            */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/



#ifndef __CALC_COLUMN_DENSITY_HPP_INCLUDED__
#define __CALC_COLUMN_DENSITY_HPP_INCLUDED__

#include "../parameters.hpp"
#include "Magritte_config.hpp"
#include "declarations.hpp"



/* calc_column_density: calculates column density for each species, ray and grid point           */
/*-----------------------------------------------------------------------------------------------*/

int calc_column_density( GRIDPOINT *gridpoint, EVALPOINT *evalpoint,
                         double *column_density, int spec );

/*-----------------------------------------------------------------------------------------------*/



/* column_density: calculates the column density for one species along one ray                   */
/*-----------------------------------------------------------------------------------------------*/

double column_density_at_point( GRIDPOINT *gridpoint, EVALPOINT *evalpoint,
                                long gridp, int spec, long ray );

/*-----------------------------------------------------------------------------------------------*/



#endif /* __CALC_COLUMN_DENSITY_HPP_INCLUDED__ */

/*-----------------------------------------------------------------------------------------------*/
