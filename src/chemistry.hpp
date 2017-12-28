/* Frederik De Ceuster - University College London & KU Leuven                                   */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/* Header for chemistry.cpp                                                                      */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/



#ifndef __CHEMISTRY_HPP_INCLUDED__
#define __CHEMISTRY_HPP_INCLUDED__



#include "declarations.hpp"



/* abundances: calculate abundances for each species at each grid point                          */
/*-----------------------------------------------------------------------------------------------*/


#if ( ON_THE_FLY )

int chemistry( CELL *cell,
               double *temperature_gas, double *temperature_dust, double *rad_surface, double *AV,
               double *column_H2, double *column_HD, double *column_C, double *column_CO );

#else

int chemistry( CELL *cell, EVALPOINT *evalpoint,
               long *key, long *raytot, long *cum_raytot,
               double *temperature_gas, double *temperature_dust, double *rad_surface, double *AV,
               double *column_H2, double *column_HD, double *column_C, double *column_CO );

#endif


/*-----------------------------------------------------------------------------------------------*/



#endif /* __CHEMISTRY_HPP_INCLUDED__ */

/*-----------------------------------------------------------------------------------------------*/
