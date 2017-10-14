/* Frederik De Ceuster - University College London & KU Leuven                                   */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/* Header for calc_LTE_populations.cpp                                                                      */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/



#ifndef __CALC_LTE_POPULATIONS_HPP_INCLUDED__
#define __CALC_LTE_POPULATIONS_HPP_INCLUDED__



#include "declarations.hpp"


/* calc_LTE_populations: Calculates the LTE level populations                                    */
/*-----------------------------------------------------------------------------------------------*/

int calc_LTE_populations( GRIDPOINT *gridpoint, double *energy, double *weight,
                          double *temperature_gas, double *pop );

/*-----------------------------------------------------------------------------------------------*/



#endif /* __CALC_LTE_POPULATIONS_HPP_INCLUDED__ */

/*-----------------------------------------------------------------------------------------------*/
