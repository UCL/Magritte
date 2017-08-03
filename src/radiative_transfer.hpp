/* Frederik De Ceuster - University College London                                               */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/* Header for radiative_transfer.cpp                                                                      */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/



#ifndef __RADIATIVE_TRANSFER_HPP_INCLUDED__
#define __RADIATIVE_TRANSFER_HPP_INCLUDED__



#include "declarations.hpp"



/* radiative_transfer: calculate mean intensity at grid point "gridp", by solving the transfer   */
/*                     equation along all pairs of a rays and their antipodals                   */
/*-----------------------------------------------------------------------------------------------*/

void radiative_transfer( long *antipod, EVALPOINT *evalpoint, double *P_intensity,
                         double *mean_intensity, double *Source, double *opacity,
                         int *irad, int*jrad, long gridp, int lspec, int kr,
                         long *nshortcuts, long *nno_shortcuts );

/*-----------------------------------------------------------------------------------------------*/



#endif /* __RADIATIVE_TRANSFER_HPP_INCLUDED__ */

/*-----------------------------------------------------------------------------------------------*/
