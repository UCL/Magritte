/* Frederik De Ceuster - University College London & KU Leuven                                   */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/* Header for cell_radiative_transfer.cpp                                                        */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/



#ifndef __CELL_RADIATIVE_TRANSFER_HPP_INCLUDED__
#define __CELL_RADIATIVE_TRANSFER_HPP_INCLUDED__



#include "../parameters.hpp"
#include "Magritte_config.hpp"
#include "declarations.hpp"



#if (CELL_BASED)

/* radiative_transfer: calculate mean intensity at grid point "gridp", by solving the transfer   */
/*                     equation along all pairs of a rays and their antipodals                   */
/*-----------------------------------------------------------------------------------------------*/

int cell_radiative_transfer( CELL *cell, double *mean_intensity, double *Lambda_diagonal,
                             double *mean_intensity_eff, double *Source, double *opacity,
                             double *frequency, double *temperature_gas, double *temperature_dust,
                             int *irad, int*jrad, long gridp, int lspec, int kr );

/*-----------------------------------------------------------------------------------------------*/



/* intensity: calculate the intensity along a certain ray through a certain point                */
/*-----------------------------------------------------------------------------------------------*/

int intensities( CELL *cell, double *source, double *opacity, double *frequency,
                 double freq, double *temperature_gas,  int *irad, int*jrad, long gridp, long r,
                 int lspec, int kr, double *u_local, double *v_local, double *L_local );

/*-----------------------------------------------------------------------------------------------*/

#endif



#endif /* __CELL_RADIATIVE_TRANSFER_HPP_INCLUDED__ */

/*-----------------------------------------------------------------------------------------------*/
