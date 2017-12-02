/* Frederik De Ceuster - University College London & KU Leuven                                   */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/* Header for initializers.cpp                                                                   */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/



#ifndef __LINE_PROFILE_HPP_INCLUDED__
#define __LINE_PROFILE_HPP_INCLUDED__



#include "../parameters.hpp"
#include "Magritte_config.hpp"
#include "declarations.hpp"



/* line_source: calculate the line source function                                               */
/*-----------------------------------------------------------------------------------------------*/

int line_source( int *irad, int *jrad, double *A_coeff, double *B_coeff, double *pop, int lspec,
                 double *source );

/*-----------------------------------------------------------------------------------------------*/



/* line_opacity: calculate the line opacity                                                      */
/*-----------------------------------------------------------------------------------------------*/

int line_opacity( int *irad, int *jrad, double *frequency, double *B_coeff, double *pop, int lspec,
                  double *opacity );

/*-----------------------------------------------------------------------------------------------*/



/* line_profile: calculate the line profile function                                             */
/*-----------------------------------------------------------------------------------------------*/

double line_profile( EVALPOINT *evalpoint, double *temperature_gas, double frequency,
                     double line_frequency, long gridp );

/*-----------------------------------------------------------------------------------------------*/



#endif /* __LINE_PROFILE_HPP_INCLUDED__ */

/*-----------------------------------------------------------------------------------------------*/
