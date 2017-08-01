/* Frederik De Ceuster - University College London                                               */
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

double exact_feautrier( long ndep, double *S, double *dtau, long etot1, long etot2,
                        EVALPOINT *evalpoint, double *P_intensity, long gridp, long r1, long ar1 );

/*-----------------------------------------------------------------------------------------------*/



#endif /* __EXACT_FEAUTRIER_HPP_INCLUDED__ */

/*-----------------------------------------------------------------------------------------------*/
