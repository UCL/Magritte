/* Frederik De Ceuster - University College London                                               */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/* Header for column_density_calculator.cpp                                                      */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/



#ifndef __COLUMN_DENSITY_CALCULATOR_HPP_INCLUDED__
#define __COLUMN_DENSITY_CALCULATOR_HPP_INCLUDED__

#include "declarations.hpp"



/* column_density_calculator: calculates column density for each species, ray and grid point     */
/*-----------------------------------------------------------------------------------------------*/

void column_density_calculator( GRIDPOINT *gridpoint, EVALPOINT *evalpoint,
                                double *column_density, double *AV );

/*-----------------------------------------------------------------------------------------------*/



#endif /* __COLUMN_DENSITY_CALCULATOR_HPP_INCLUDED__ */

/*-----------------------------------------------------------------------------------------------*/
