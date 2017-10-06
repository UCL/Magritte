/* Frederik De Ceuster - University College London & KU Leuven                                   */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/* Header for exact_feautrier.cpp                                                                */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/



#ifndef __EXACT_FEAUTRIER_HPP_INCLUDED__
#define __EXACT_FEAUTRIER_HPP_INCLUDED__



#include "declarations.hpp"



/* exact_feautrier: fill Feautrier matrix, solve exactly, return (P[etot1-1]+P[etot1-2])/2       */
/*-----------------------------------------------------------------------------------------------*/

int exact_feautrier( long ndep, double *S, double *dtau, long etot1, long etot2, double ibc,
                     EVALPOINT *evalpoint, double *P_intensity, long gridp, long r, long ar );

/*-----------------------------------------------------------------------------------------------*/



#endif /* __EXACT_FEAUTRIER_HPP_INCLUDED__ */

/*-----------------------------------------------------------------------------------------------*/
