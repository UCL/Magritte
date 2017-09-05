/* Frederik De Ceuster - University College London                                               */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/* Header for write_output.cpp                                                                   */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/



#ifndef __WRITE_OUTPUT_HPP_INCLUDED__
#define __WRITE_OUTPUT_HPP_INCLUDED__

#include "declarations.hpp"



/* writing_output: write the output files
/*-----------------------------------------------------------------------------------------------*/

void write_output( double *unit_healpixvector, long *antipod,
                   GRIDPOINT *gridpoint, EVALPOINT *evalpoint,
                   double *pop, double *weight, double *energy, double *mean_intensity,
                   double *temperature_gas, double *temperature_dust );

/*-----------------------------------------------------------------------------------------------*/



#endif /* __WRITE_OUTPUT_HPP_INCLUDED__ */

/*-----------------------------------------------------------------------------------------------*/
