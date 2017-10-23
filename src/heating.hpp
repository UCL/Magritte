/* Frederik De Ceuster - University College London & KU Leuven                                   */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/* Header for heating.cpp                                                                        */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/



#ifndef __HEATING_HPP_INCLUDED__
#define __HEATING_HPP_INCLUDED__

#include "declarations.hpp"



/* heating: calculate the total heating                                                          */
/*-----------------------------------------------------------------------------------------------*/

double heating( GRIDPOINT *gridpoint, long gridp,
                double *temperature_gas, double *temperature_dust,
                double *UV_field, double v_turb, double* heating_components );

/*-----------------------------------------------------------------------------------------------*/



/* F: mathematical function used in photoelectric dust heating                                   */
/*-----------------------------------------------------------------------------------------------*/

double F(double x, double delta, double gamma);

/*-----------------------------------------------------------------------------------------------*/



/* dF: defivative w.r.t. x of the function F defined above                                       */
/*-----------------------------------------------------------------------------------------------*/

double dF(double x, double delta);

/*-----------------------------------------------------------------------------------------------*/



#endif /* __HEATING_HPP_INCLUDED__ */

/*-----------------------------------------------------------------------------------------------*/
