/* Frederik De Ceuster - University College London                                               */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/* Header for calc_reac_rates_rad.cpp                                                     */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/



#ifndef __calc_reac_rates_rad_HPP_INCLUDED__
#define __calc_reac_rates_rad_HPP_INCLUDED__



/* Note in the arguments that the temperatures are local (doubles), but rad_surface, AV and column
   densities are still represented by the pointers to the full arrays */



/* rate_PHOTD: returns rate coefficient for photodesorption                                      */
/*-----------------------------------------------------------------------------------------------*/

double rate_PHOTD( int reac, double temperature_gas, double *rad_surface, double *AV, long gridp );

/*-----------------------------------------------------------------------------------------------*/



/* rate_H2_photodissociation: returns rate coefficient for H2 dissociation                       */
/*-----------------------------------------------------------------------------------------------*/

double rate_H2_photodissociation( int reac, double *rad_surface,
                                  double *AV, double *column_H2, double v_turb, long gridp );

/*-----------------------------------------------------------------------------------------------*/



/* rate_CO_photodissociation: returns rate coefficient for CO dissociation                       */
/*-----------------------------------------------------------------------------------------------*/

double rate_CO_photodissociation( int reac, double *rad_surface,
                                  double *AV, double *column_CO, double *column_H2, long gridp );

/*-----------------------------------------------------------------------------------------------*/



/* rate_C_photoionization: returns rate coefficient for C photoionization                        */
/*-----------------------------------------------------------------------------------------------*/

double rate_C_photoionization( int reac, double temperature_gas,
                               double *rad_surface, double *AV,
                               double *column_C, double *column_H2, long gridp );

/*-----------------------------------------------------------------------------------------------*/



/* rate_SI_photoionization: returns rate coefficient for SI photoionization                      */
/*-----------------------------------------------------------------------------------------------*/

double rate_SI_photoionization( int reac, double *rad_surface, double *AV, long gridp );

/*-----------------------------------------------------------------------------------------------*/



#endif /* __calc_reac_rates_rad_HPP_INCLUDED__ */

/*-----------------------------------------------------------------------------------------------*/
