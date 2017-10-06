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

int update_temperature_gas( double thermal_flux, long gridp, double *temperature_gas,
                            double* previous_temperature_gas );

/*-----------------------------------------------------------------------------------------------*/



#endif /* __UPDATE_TEMPERATURE_GAS_HPP_INCLUDED__ */

/*-----------------------------------------------------------------------------------------------*/
