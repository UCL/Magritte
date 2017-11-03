/* Frederik De Ceuster - University College London & KU Leuven                                   */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/* Header for calc_column_density.cpp                                                      */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/



#ifndef __calc_AV_HPP_INCLUDED__
#define __calc_AV_HPP_INCLUDED__

#include "../parameters.hpp"
#include "Magritte_config.hpp"
#include "declarations.hpp"



/* calc_AV: calculates the visual extinction along a ray ray at a grid point               */
/*-----------------------------------------------------------------------------------------------*/

int calc_AV( double *column_tot, double *AV );

/*-----------------------------------------------------------------------------------------------*/



#endif /* __calc_AV_HPP_INCLUDED__ */

/*-----------------------------------------------------------------------------------------------*/
