/* Frederik De Ceuster - University College London & KU Leuven                                   */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/* Header for feautrier.cpp                                                                      */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/



#ifndef __feautrier_HPP_INCLUDED__
#define __feautrier_HPP_INCLUDED__



#include "../parameters.hpp"
#include "Magritte_config.hpp"
#include "declarations.hpp"



/* feautrier: fill Feautrier matrix, solve exactly, return (P[etot1-1]+P[etot1-2])/2             */
/*-----------------------------------------------------------------------------------------------*/

int feautrier( EVALPOINT *evalpoint, long *key, long *raytot, long *cum_raytot, long gridp, long r,
               double *S, double *dtau, double *u_intensity, double *L_diag_approx );

/*-----------------------------------------------------------------------------------------------*/



#endif /* __feautrier_HPP_INCLUDED__ */

/*-----------------------------------------------------------------------------------------------*/
