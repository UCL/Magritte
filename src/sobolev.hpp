/* Frederik De Ceuster - University College London & KU Leuven                                   */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/* Header for sobolev.cpp                                                                        */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/



#ifndef __SOBOLEV_HPP_INCLUDED__
#define __SOBOLEV_HPP_INCLUDED__



#include "../parameters.hpp"
#include "Magritte_config.hpp"
#include "declarations.hpp"



/* feautrier: fill Feautrier matrix, solve exactly, return (P[etot1-1]+P[etot1-2])/2             */
/*-----------------------------------------------------------------------------------------------*/

int sobolev( GRIDPOINT *gridpoint, EVALPOINT *evalpoint, long *key, long *raytot, long *cum_raytot,
             double *mean_intensity, double *Lambda_diagonal, double *mean_intensity_eff,
             double *source, double *opacity, double *frequency, double *temperature_gas,
             double *temperature_dust, int *irad, int*jrad, long gridp, int lspec, int kr );

/*-----------------------------------------------------------------------------------------------*/



#endif /* __SOBOLEV_HPP_INCLUDED__ */

/*-----------------------------------------------------------------------------------------------*/
