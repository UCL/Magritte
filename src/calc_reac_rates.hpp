/* Frederik De Ceuster - University College London                                               */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/* Header for calc_reac_rates.cpp                                                              */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/



#ifndef __calc_reac_rates_HPP_INCLUDED__
#define __calc_reac_rates_HPP_INCLUDED__



/* rate_H2_formation: returns the rate coefficient for the H2 formation reaction                 */
/*-----------------------------------------------------------------------------------------------*/

double rate_H2_formation( int reac, double temperature_gas, double temperature_dust );

/*-----------------------------------------------------------------------------------------------*/


/* rate_PAH: returns the rate coefficient for the reactions with PAHs                            */
/*-----------------------------------------------------------------------------------------------*/

double rate_PAH( int reac, double temperature_gas );

/*-----------------------------------------------------------------------------------------------*/



/* rate_CRP: returns the rate coefficient for the reaction induced by cosmic rays                */
/*-----------------------------------------------------------------------------------------------*/

double rate_CRP( int reac, double temperature_gas );

/*-----------------------------------------------------------------------------------------------*/



/* rate_CRPHOT: returns the rate coefficient for the reaction induced by cosmic rays             */
/*-----------------------------------------------------------------------------------------------*/

double rate_CRPHOT( int reac, double temperature_gas );

/*-----------------------------------------------------------------------------------------------*/



/* rate_FREEZE: returns the rate coefficient for freeze-out reaction of neutral species          */
/*-----------------------------------------------------------------------------------------------*/

double rate_FREEZE( int reac, double temperature_gas );

/*-----------------------------------------------------------------------------------------------*/



/* rate_ELFRZE: returns rate coefficient for freeze-out reaction of singly charged positive ions */
/*-----------------------------------------------------------------------------------------------*/

double rate_ELFRZE( int reac, double temperature_gas );

/*-----------------------------------------------------------------------------------------------*/



/* rate_CRH: returns rate coefficient for desorption due to cosmic ray heating                   */
/*-----------------------------------------------------------------------------------------------*/

double rate_CRH( int reac, double temperature_gas );

/*-----------------------------------------------------------------------------------------------*/



/* rate_THERM: returns rate coefficient for thermal desorption                                   */
/*-----------------------------------------------------------------------------------------------*/

double rate_THERM( int reac, double temperature_gas, double temperature_dust );

/*-----------------------------------------------------------------------------------------------*/



/* rate_GM: returns rate coefficient for grain mantle reactions                                  */
/*-----------------------------------------------------------------------------------------------*/

double rate_GM( int reac );

/*-----------------------------------------------------------------------------------------------*/



/* rate_canonical: returns the canonical rate coefficient for the reaction                       */
/*-----------------------------------------------------------------------------------------------*/

double rate_canonical( int reac, double temperature_gas );

/*-----------------------------------------------------------------------------------------------*/



#endif /* __calc_reac_rates_HPP_INCLUDED__ */

/*-----------------------------------------------------------------------------------------------*/
