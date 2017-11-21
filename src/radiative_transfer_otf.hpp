/* Frederik De Ceuster - University College London & KU Leuven                                   */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/* Header for radiative_transfer.cpp                                                             */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/



#ifndef __RADIATIVE_TRANSFER_OTF_HPP_INCLUDED__
#define __RADIATIVE_TRANSFER_OTF_HPP_INCLUDED__



#include "../parameters.hpp"
#include "Magritte_config.hpp"
#include "declarations.hpp"



#ifdef ON_THE_FLY

/* radiative_transfer: calculate mean intensity at grid point "gridp", by solving the transfer   */
/*                     equation along all pairs of a rays and their antipodals                   */
/*-----------------------------------------------------------------------------------------------*/

void radiative_transfer_otf( GRIDPOINT *gridpoint, EVALPOINT *evalpoint,
                             long *key, long *raytot, long *cum_raytot,
                             double *mean_intensity,
                             double *Lambda_diagonal, double *mean_intensity_eff,
                             double *Source, double *opacity, double *frequency,
                             double *temperature_gas, double *temperature_dust,
                             int *irad, int*jrad, long gridp, int lspec, int kr );

/*-----------------------------------------------------------------------------------------------*/

#endif



#endif /* __RADIATIVE_TRANSFER_OTF_HPP_INCLUDED__ */

/*-----------------------------------------------------------------------------------------------*/
