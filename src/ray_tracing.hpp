/* Frederik De Ceuster - University College London & KU Leuven                                   */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/* Header for ray_tracing.cpp                                                                    */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/



#ifndef __RAY_TRACING_HPP_INCLUDED__
#define __RAY_TRACING_HPP_INCLUDED__



#include "../parameters.hpp"
#include "Magritte_config.hpp"
#include "declarations.hpp"



#if !( ON_THE_FLY )

/* ray_tracing: creates the evaluation points for each ray for each grid point                   */
/*-----------------------------------------------------------------------------------------------*/

int ray_tracing( GRIDPOINT *gridpoint, EVALPOINT *evalpoint,
                 long *key, long *raytot, long *cum_raytot );

/*-----------------------------------------------------------------------------------------------*/



#else

/* get_evalpoints: creates the evaluation points for each ray for this grid point                */
/*-----------------------------------------------------------------------------------------------*/

int get_local_evalpoint( GRIDPOINT *gridpoint, EVALPOINT *evalpoint,
                         long *key, long *raytot, long *cum_raytot, long gridp );

/*-----------------------------------------------------------------------------------------------*/



/* get_velocities: get the velocity of the evaluation point with respect to the grid point       */
/*-----------------------------------------------------------------------------------------------*/

int get_velocities( GRIDPOINT *gridpoint, EVALPOINT *evalpoint,
                    long *key, long *raytot, long *cum_raytot, long gridp, long *first_velo );

/*-----------------------------------------------------------------------------------------------*/

#endif



/* find_neighbors: creates the evaluation points for each ray for each cell                      */
/*-----------------------------------------------------------------------------------------------*/

int find_neighbors( long ncells, CELL *cell );

/*-----------------------------------------------------------------------------------------------*/



/* next_cell_on_ray: find the number of the next cell on the ray and its distance along the ray  */
/*-----------------------------------------------------------------------------------------------*/

int next_cell_on_ray( long ncells, CELL *cell, long current, long origin, long ray, double Z,
                      long *next, double *dZ );

/*-----------------------------------------------------------------------------------------------*/



#endif /* __RAY_TRACING_HPP_INCLUDED__ */

/*-----------------------------------------------------------------------------------------------*/
