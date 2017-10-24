/* Frederik De Ceuster - University College London & KU Leuven                                   */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/* Header for reaction_rates.cpp                                                                      */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/



#ifndef __REACTION_RATES_HPP_INCLUDED__
#define __REACTION_RATES_HPP_INCLUDED__



/* reaction_rates: Check which kind of reaction and call appropriate rate calculator b           */
/*-----------------------------------------------------------------------------------------------*/

void reaction_rates( double *temperature_gas, double *temperature_dust,
                     double *rad_surface, double *AV,
                     double *column_H2, double *column_HD, double *column_C, double *column_CO,
                     double v_turb, long gridp );

/*-----------------------------------------------------------------------------------------------*/



#endif /* __REACTION_RATES_HPP_INCLUDED__ */

/*-----------------------------------------------------------------------------------------------*/
