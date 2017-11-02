/* Frederik De Ceuster - University College London & KU Leuven                                   */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/* Header for update_temperature_gas.cpp                                                         */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/



#ifndef __UPDATE_TEMPERATURE_GAS_HPP_INCLUDED__
#define __UPDATE_TEMPERATURE_GAS_HPP_INCLUDED__



/* update_temperature_gas: update the gas temperature after a thermal balance iteration          */
/*-----------------------------------------------------------------------------------------------*/

int update_temperature_gas( double *thermal_ratio, double* thermal_sum, long gridp, double *temperature_gas,
                            double *previous_temperature_gas,
                            double *temperature_a, double *temperature_b,
                            double *thermal_ratio_a, double *thermal_ratio_b );

/*-----------------------------------------------------------------------------------------------*/



/* shuffle_temperatures: rename the variables for Brent's method                                 */
/*-----------------------------------------------------------------------------------------------*/

int shuffle_Brent( long gridp, double *temperature_a, double *temperature_b, double *temperature_c,
                   double *temperature_d, double *temperature_e, double *thermal_ratio_a,
                   double *thermal_ratio_b, double *thermal_ratio_c );

/*-----------------------------------------------------------------------------------------------*/



/* update_temperature_gas: update the gas temperature using Brent's method                       */
/*-----------------------------------------------------------------------------------------------*/

int update_temperature_gas_Brent( long gridp, double *temperature_a, double *temperature_b,
                                  double *temperature_c, double *temperature_d,
                                  double *temperature_e, double *thermal_ratio_a,
                                  double *thermal_ratio_b, double *thermal_ratio_c );

/*-----------------------------------------------------------------------------------------------*/



#endif /* __UPDATE_TEMPERATURE_GAS_HPP_INCLUDED__ */

/*-----------------------------------------------------------------------------------------------*/
