/* Frederik De Ceuster - University College London                                               */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/* Header for rate_calculations_radfield.cpp                                                     */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/



#ifndef __RATE_CALCULATIONS_RADFIELD_HPP_INCLUDED__
#define __RATE_CALCULATIONS_RADFIELD_HPP_INCLUDED__



/* rate_PHOTD: returns rate coefficient for photodesorption                                      */
/*-----------------------------------------------------------------------------------------------*/

double rate_PHOTD( int reac, double temperature_gas, double *rad_surface, double *AV );

/*-----------------------------------------------------------------------------------------------*/



/* rate_H2_photodissociation: returns rate coefficient for H2 dissociation                       */
/*-----------------------------------------------------------------------------------------------*/

double rate_H2_photodissociation( int reac, double *rad_surface,
                                  double *AV, double *column_H2, double v_turb );

/*-----------------------------------------------------------------------------------------------*/



/* rate_CO_photodissociation: returns rate coefficient for CO dissociation                       */
/*-----------------------------------------------------------------------------------------------*/

double rate_CO_photodissociation( int reac, double *rad_surface,
                                  double *AV, double *column_CO, double *column_H2 );

/*-----------------------------------------------------------------------------------------------*/



/* rate_CI_photoionization: returns rate coefficient for CI photoionization                      */
/*-----------------------------------------------------------------------------------------------*/

double rate_CI_photoionization( int reac, double temperature_gas,
                                double *rad_surface, double *AV,
                                double *column_CI, double *column_H2 );

/*-----------------------------------------------------------------------------------------------*/



/* rate_SI_photoionization: returns rate coefficient for SI photoionization                      */
/*-----------------------------------------------------------------------------------------------*/

double rate_SI_photoionization( int reac, double *rad_surface, double *AV );

/*-----------------------------------------------------------------------------------------------*/



/* self_shielding_H2: Returns H2 self-shielding function                                         */
/*-----------------------------------------------------------------------------------------------*/

double self_shielding_H2( double column_H2, double doppler_width, double radiation_width );

/*-----------------------------------------------------------------------------------------------*/



/* self_shielding_CO: Returns CO self-shielding function                                         */
/*-----------------------------------------------------------------------------------------------*/

double self_shielding_CO( double column_CO, double column_H2 );

/*-----------------------------------------------------------------------------------------------*/



/* dust_scattering: Retuns the attenuation due to scattering by dust                             */
/*-----------------------------------------------------------------------------------------------*/

double dust_scattering( double AV_ray, double lambda );

/*-----------------------------------------------------------------------------------------------*/



/* X_lambda: Retuns ratio of optical depths at given lambda w.r.t. the visual wavelenght         */
/*-----------------------------------------------------------------------------------------------*/

double X_lambda(double lambda);

/*-----------------------------------------------------------------------------------------------*/



#endif /* __RATE_CALCULATIONS_RADFIELD_HPP_INCLUDED__ */

/*-----------------------------------------------------------------------------------------------*/
