/* Frederik De Ceuster - University College London & KU Leuven                                   */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/* Header for feautrier.cpp                                                                */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/



#ifndef __feautrier_HPP_INCLUDED__
#define __feautrier_HPP_INCLUDED__



#include "../parameters.hpp"
#include "Magritte_config.hpp"
#include "declarations.hpp"



/* feautrier: fill Feautrier matrix, solve exactly, return (P[etot1-1]+P[etot1-2])/2       */
/*-----------------------------------------------------------------------------------------------*/

int feautrier( EVALPOINT *evalpoint, long gridp, long r, long ar, double *S, double *dtau,
               double *P_intensity, double *Lambda_diagonal );

/*-----------------------------------------------------------------------------------------------*/



#endif /* __feautrier_HPP_INCLUDED__ */

/*-----------------------------------------------------------------------------------------------*/
