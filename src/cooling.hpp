/* Frederik De Ceuster - University College London & KU Leuven                                   */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/* Header for heating.cpp                                                                        */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/



#ifndef __COOLING_HPP_INCLUDED__
#define __COOLING_HPP_INCLUDED__



#include "../parameters.hpp"
#include "Magritte_config.hpp"
#include "declarations.hpp"



/* cooling: calculate the total cooling                                                          */
/*-----------------------------------------------------------------------------------------------*/

double cooling( long gridp, int *irad, int *jrad, double *A_coeff, double *B_coeff,
                double *frequency, double *weight, double *pop, double *mean_intensity );

/*-----------------------------------------------------------------------------------------------*/




#endif /* __COOLING_HPP_INCLUDED__ */

/*-----------------------------------------------------------------------------------------------*/
