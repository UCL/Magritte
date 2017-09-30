/* Frederik De Ceuster - University College London                                               */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/* Header for calc_column_density.cpp                                                      */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/



#ifndef __calc_column_density_HPP_INCLUDED__
#define __calc_column_density_HPP_INCLUDED__

#include "declarations.hpp"



/* calc_column_density: calculates column density for each species, ray and grid point     */
/*-----------------------------------------------------------------------------------------------*/

void calc_column_density( GRIDPOINT *gridpoint, EVALPOINT *evalpoint,
                                double *column_density, int spec );

/*-----------------------------------------------------------------------------------------------*/



/* column_density: calculates the column density for one species along one ray                   */
/*-----------------------------------------------------------------------------------------------*/

double column_density_at_point( GRIDPOINT *gridpoint, EVALPOINT *evalpoint,
                                long gridp, int spec, long ray );

/*-----------------------------------------------------------------------------------------------*/



#endif /* __calc_column_density_HPP_INCLUDED__ */

/*-----------------------------------------------------------------------------------------------*/
