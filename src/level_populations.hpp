/* Frederik De Ceuster - University College London                                               */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/* Header for level_populations.cpp                                                                      */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/



#ifndef __LEVEL_POPULATIONS_HPP_INCLUDED__
#define __LEVEL_POPULATIONS_HPP_INCLUDED__



#include "declarations.hpp"



/* level_populations: iteratively calculates the level populations                               */
/*-----------------------------------------------------------------------------------------------*/

void level_populations( long *antipod, GRIDPOINT *gridpoint, EVALPOINT *evalpoint,
                        int *irad, int*jrad, double *frequency, double *A_coeff,
                        double *B_coeff, double *C_coeff, double *P_intensity,
                        double *R, double *pop, double *dpop, double *C_data,
                        double *coltemp, int *icol, int *jcol, double *temperature_gas,
                        double *weight, double *energy, int lspec );

/*-----------------------------------------------------------------------------------------------*/



#endif /* __LEVEL_POPULATIONS_HPP_INCLUDED__ */

/*-----------------------------------------------------------------------------------------------*/
