/* Frederik De Ceuster - University College London                                               */
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
                     double v_turb, long gridp,
                     int *nr_can_reac, int *canonical_reactions,
                     int *nr_can_phot, int *can_photo_reactions,
                     int *nr_all, int *all_reactions );

/*-----------------------------------------------------------------------------------------------*/



#endif /* __REACTION_RATES_HPP_INCLUDED__ */

/*-----------------------------------------------------------------------------------------------*/
