/* Frederik De Ceuster - University College London & KU Leuven                                   */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/* Header for level_populations.cpp                                                              */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/



#ifndef __LEVEL_POPULATIONS_HPP_INCLUDED__
#define __LEVEL_POPULATIONS_HPP_INCLUDED__



#include "../parameters.hpp"
#include "Magritte_config.hpp"
#include "declarations.hpp"



/* level_populations: iteratively calculates the level populations                               */
/*-----------------------------------------------------------------------------------------------*/

int level_populations( GRIDPOINT *gridpoint, EVALPOINT *evalpoint,
                       int *irad, int*jrad, double *frequency,
                       double *A_coeff, double *B_coeff, double *C_coeff, double *R,
                       double *pop, double *prev1_pop, double *prev2_pop, double *prev3_pop,
                       double *C_data, double *coltemp, int *icol, int *jcol,
                       double *temperature_gas, double *temperature_dust,
                       double *weight, double *energy, double *mean_intensity,
                       double *Lambda_diagonal, double *mean_intensity_eff );

/*-----------------------------------------------------------------------------------------------*/



#endif /* __LEVEL_POPULATIONS_HPP_INCLUDED__ */

/*-----------------------------------------------------------------------------------------------*/
