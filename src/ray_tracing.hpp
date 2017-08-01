/* Frederik De Ceuster - University College London                                               */
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



#include "declarations.hpp"



/* ray_tracing: creates the evaluation points for each ray for each grid point                   */
/*-----------------------------------------------------------------------------------------------*/

void ray_tracing( double theta_crit, double ray_separation2, double *unit_healpixvector,
                  GRIDPOINT *gridpoint, EVALPOINT *evalpoint );

/*-----------------------------------------------------------------------------------------------*/



#endif /* __RAY_TRACING_HPP_INCLUDED__ */

/*-----------------------------------------------------------------------------------------------*/
