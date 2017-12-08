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


#if ( ON_THE_FLY )

int calc_column_density( GRIDPOINT *gridpoint, double *column_density, int spec );

#else

int calc_column_density( GRIDPOINT *gridpoint, EVALPOINT *evalpoint, long *key, long *raytot,
                         long *cum_raytot, double *column_density, int spec );

#endif


/*-----------------------------------------------------------------------------------------------*/


#if ( ON_THE_FLY )

/* calc_column_densities: calculates column densities for the species needed in chemistry        */
/*-----------------------------------------------------------------------------------------------*/

int calc_column_densities( GRIDPOINT *gridpoint, double *column_H2, double *column_HD,
                           double *column_C, double *column_CO );

/*-----------------------------------------------------------------------------------------------*/

#endif


/* column_density: calculates the column density for one species along one ray                   */
/*-----------------------------------------------------------------------------------------------*/

double column_density_at_point( GRIDPOINT *gridpoint, EVALPOINT *evalpoint, long *key,
                                long *raytot, long *cum_raytot, long gridp, int spec, long ray );

/*-----------------------------------------------------------------------------------------------*/



#endif /* __CALC_COLUMN_DENSITY_HPP_INCLUDED__ */

/*-----------------------------------------------------------------------------------------------*/
