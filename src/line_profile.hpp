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



#include "declarations.hpp"



/* line_profile: calculate the line profile function                                             */
/*-----------------------------------------------------------------------------------------------*/

double line_profile( EVALPOINT evalpoint, double *temperature_gas, double frequency,
                     long gridp, long evalp );

/*-----------------------------------------------------------------------------------------------*/



#endif /* __LINE_PROFILE_HPP_INCLUDED__ */

/*-----------------------------------------------------------------------------------------------*/
