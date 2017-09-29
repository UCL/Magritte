/* Frederik De Ceuster - University College London                                               */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/* Header for abundance.cpp                                                                      */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/



#ifndef __ABUNDANCES_HPP_INCLUDED__
#define __ABUNDANCES_HPP_INCLUDED__



/* abundances: calculate abundances for each species at each grid point                          */
/*-----------------------------------------------------------------------------------------------*/

int chemistry( GRIDPOINT *gridpoint, double *temperature_gas, double *temperature_dust,
                double *rad_surface, double *AV,
                double *column_H2, double *column_HD, double *column_C, double *column_CO,
                double v_turb );

/*-----------------------------------------------------------------------------------------------*/



#endif /* __ABUNDANCES_HPP_INCLUDED__ */

/*-----------------------------------------------------------------------------------------------*/
