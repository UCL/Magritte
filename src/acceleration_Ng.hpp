/* Frederik De Ceuster - University College London & KU Leuven                                   */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/* Header for acceleration_Ng.cpp                                                                */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/



#ifndef __ACCELERATION_NG_HPP_INCLUDED__
#define __ACCELERATION_NG_HPP_INCLUDED__



/* acceleration_Ng: perform a Ng accelerated iteration for the level populations                 */
/*-----------------------------------------------------------------------------------------------*/

int acceleration_Ng( int lspec, double *prev3_pop, double *prev2_pop, double *prev1_pop,
                     double *pop );

/*-----------------------------------------------------------------------------------------------*/



/* store_populations: update the previous populations                                            */
/*-----------------------------------------------------------------------------------------------*/

int store_populations( int lspec, double *prev3_pop, double *prev2_pop, double *prev1_pop,
                       double *pop );

/*-----------------------------------------------------------------------------------------------*/



/* acceleration_temperature_Ng: perform a Ng accelerated iteration for the gas temperature       */
/*-----------------------------------------------------------------------------------------------*/

int acceleration_temperature_Ng( double *prev3_temperature_gas, double *prev2_temperature_gas,
                                 double *prev1_temperature_gas, double *temperature_gas );


/*-----------------------------------------------------------------------------------------------*/



/* store_temperatures: update the previous temperatures                                          */
/*-----------------------------------------------------------------------------------------------*/

int store_temperatures( double *prev3_temperature_gas, double *prev2_temperature_gas,
                        double *prev1_temperature_gas, double *temperature_gas );

/*-----------------------------------------------------------------------------------------------*/



#endif /* __ACCELERATION_NG_HPP_INCLUDED__ */

/*-----------------------------------------------------------------------------------------------*/
