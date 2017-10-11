/* Frederik De Ceuster - University College London & KU Leuven                                   */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/* calc_reac_rates_rad: Calculate rates for reactions depending on the radiation field    */
/*                                                                                               */
/* (based on H2_form, shield and calc_reac_rates in 3D-PDR)                                      */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>

#include "declarations.hpp"
#include "calc_reac_rates_rad.hpp"
#include "radfield_tools.hpp"
#include "data_tools.hpp"



/* Note in the arguments that the temperatures are local (doubles), but rad_surface, AV and column
   densities are still represented by the pointers to the full arrays */



/* rate_PHOTD: returns rate coefficient for photodesorption                                      */
/*-----------------------------------------------------------------------------------------------*/

double rate_PHOTD( int reac, double temperature_gas, double *rad_surface, double *AV, long gridp )
{


  double yield;                   /* Number of adsorbed molecules released per cosmic ray impact */

  double flux;                                                /* Flux of photons (in cm^-2 s^-1) */

  // flux = 1.0E8; /* Flux of FUV photons in the unattenuated Habing field ( photons cm^-2 s^-1) */

  flux = 1.7E8;    /* Flux of FUV photons in the unattenuated Draine field ( photons cm^-2 s^-1) */

  double grain_param = 2.4E-22;   /* <d_g a^2> average grain density times radius squared (cm^2) */
                                      /* = average grain surface area per H atom (devided by PI) */

  double alpha = reaction[reac].alpha;
  double beta  = reaction[reac].beta;
  double gamma = reaction[reac].gamma;

  double RT_min = reaction[reac].RT_min;
  double RT_max = reaction[reac].RT_max;

  double k = 0.0;                                                        /* reaction coefficient */


  if ( temperature_gas < 50.0 ){

    yield = 3.5E-3;
  }

  else if ( temperature_gas < 85.0){

    yield = 4.0E-3;
  }

  else if ( temperature_gas < 100.0 ){

    yield = 5.5E-3;
  }

  else {

    yield = 7.5E-3;
  }


  /* For all rays */

  for (long ray=0; ray<NRAYS; ray++){

    k = k + flux * rad_surface[RINDEX(gridp,ray)]
                 * exp(-1.8*AV[RINDEX(gridp,ray)]) * grain_param * yield;
  }

  return k;
}

/*-----------------------------------------------------------------------------------------------*/





/* rate_H2_photodissociation: returns rate coefficient for H2 dissociation                       */
/*-----------------------------------------------------------------------------------------------*/

double rate_H2_photodissociation( int reac, double *rad_surface,
                                  double *AV, double *column_H2, double v_turb, long gridp )
{


  double alpha = reaction[reac].alpha;
  double beta  = reaction[reac].beta;
  double gamma = reaction[reac].gamma;

  double RT_min = reaction[reac].RT_min;
  double RT_max = reaction[reac].RT_max;

  double k = 0.0;                                                        /* reaction coefficient */


  double lambda = 1000.0;                           /* wavelength (in Å) of a typical transition */

  double doppler_width = v_turb / (lambda*1.0E-8);    /* linewidth (in Hz) of typical transition */
                                                /* (assuming turbulent broadening with b=3 km/s) */

  double radiation_width = 8.0E7;         /* radiative linewidth (in Hz) of a typical transition */

  cout << "v turb : " << v_turb << "\n";

  for (long ray=0; ray<NRAYS; ray++){

    k = k + alpha * rad_surface[RINDEX(gridp,ray)]
            * self_shielding_H2( column_H2[RINDEX(gridp,ray)], doppler_width, radiation_width )
            * dust_scattering( AV[RINDEX(gridp,ray)], lambda ) / 2.0;
  }

  return k;
}
/*-----------------------------------------------------------------------------------------------*/





/* rate_CO_photodissociation: returns rate coefficient for CO dissociation                       */
/*-----------------------------------------------------------------------------------------------*/

double rate_CO_photodissociation( int reac, double *rad_surface,
                                  double *AV, double *column_CO, double *column_H2, long gridp )
{


  double alpha = reaction[reac].alpha;
  double beta  = reaction[reac].beta;
  double gamma = reaction[reac].gamma;

  double RT_min = reaction[reac].RT_min;
  double RT_max = reaction[reac].RT_max;

  double k = 0.0;                                                        /* reaction coefficient */


  /* Now lambda is calculated */

  for (long ray=0; ray<NRAYS; ray++){


    /* Calculate the mean wavelength (in Å) of the 33 dissociating bands,
       weighted by their fractional contribution to the total shielding
       van Dishoeck & Black (1988, ApJ, 334, 771, Equation 4) */

    double u = log10(1.0 + column_CO[RINDEX(gridp,ray)]);
    double w = log10(1.0 + column_H2[RINDEX(gridp,ray)]);


    /* mean wavelength (in Å) of 33 dissociating bands weighted
       by their fractional contribution to the total shielding */

    double lambda = (5675.0 - 200.6*w) - (571.6 - 24.09*w)*u + (18.22 - 0.7664*w)*u*u;


    /* lambda cannot be larger than the wavelength of band 33 (1076.1Å)
       and cannot be smaller than the wavelength of band 1 (913.6Å) */

    if ( lambda > 1076.1 ) {

      lambda = 1076.1;
    }

    if ( lambda < 913.6 ){

      lambda = 913.6;
    }


    k = k + alpha * rad_surface[RINDEX(gridp,ray)]
            * self_shielding_CO( column_CO[RINDEX(gridp,ray)], column_H2[RINDEX(gridp,ray)] )
            * dust_scattering( AV[RINDEX(gridp,ray)], lambda ) / 2.0;
  }


  return k;
}

/*-----------------------------------------------------------------------------------------------*/





/* rate_C_photoionization: returns rate coefficient for C photoionization                      */
/*-----------------------------------------------------------------------------------------------*/

double rate_C_photoionization( int reac, double temperature_gas,
                               double *rad_surface, double *AV,
                               double *column_C, double *column_H2, long gridp )
{


  double alpha = reaction[reac].alpha;
  double beta  = reaction[reac].beta;
  double gamma = reaction[reac].gamma;

  double RT_min = reaction[reac].RT_min;
  double RT_max = reaction[reac].RT_max;

  double k = 0.0;                                                        /* reaction coefficient */


  for (long ray=0; ray<NRAYS; ray++){


    /* Calculate the optical depth in the C absorption band, accounting
       for grain extinction and shielding by C and overlapping H2 lines */

    double tau_C = gamma*AV[RINDEX(gridp,ray)] + 1.1E-17*column_C[RINDEX(gridp,ray)]
                   + ( 0.9*pow(temperature_gas,0.27)
                          * pow(column_H2[RINDEX(gridp,ray)]/1.59E21, 0.45) );


    /* Calculate the C photoionization rate */

    k = k + alpha * rad_surface[RINDEX(gridp,ray)] * exp(-tau_C) / 2.0;
  }

  return k;
}

/*-----------------------------------------------------------------------------------------------*/





/* rate_SI_photoionization: returns rate coefficient for SI photoionization                      */
/*-----------------------------------------------------------------------------------------------*/

double rate_SI_photoionization( int reac, double *rad_surface, double *AV, long gridp )
{


  double alpha = reaction[reac].alpha;
  double beta  = reaction[reac].beta;
  double gamma = reaction[reac].gamma;

  double RT_min = reaction[reac].RT_min;
  double RT_max = reaction[reac].RT_max;


  double k = 0.0;                                                        /* reaction coefficient */


  for (long ray=0; ray<NRAYS; ray++){


    /* Calculate the optical depth in the SI absorption band, accounting
       for grain extinction and shielding by ??? */

    double tau_S = gamma*AV[RINDEX(gridp,ray)];


    /* Calculate the SI photoionization rate */

    k = k + alpha * rad_surface[RINDEX(gridp,ray)] * exp(-tau_S) / 2.0;
  }

  return k;
}

/*-----------------------------------------------------------------------------------------------*/





/* rate_canonical_photoreaction: returns rate coefficient for a canonical photoreaction          */
/*-----------------------------------------------------------------------------------------------*/

double rate_canonical_photoreaction( int reac, double temperature_gas, double *rad_surface,
                                     double *AV, long gridp )
{


  double alpha = reaction[reac].alpha;
  double beta  = reaction[reac].beta;
  double gamma = reaction[reac].gamma;

  double RT_min = reaction[reac].RT_min;
  double RT_max = reaction[reac].RT_max;

  double k = 0.0;                                                        /* reaction coefficient */


  /* Check for large negative gamma values that might cause discrepant
     rates at low temperatures. Set these rates to zero when T < RTMIN. */

  if ( gamma < -200.0  &&  temperature_gas < RT_min ){

    return k = 0.0;
  }

  else if ( ( (temperature_gas <= RT_max) || (RT_max == 0.0) )
            && no_better_data(reac, reaction, temperature_gas) ){

    for (long ray=0; ray<NRAYS; ray++){

      double tau = gamma*AV[RINDEX(gridp,ray)];

      k = k + alpha * rad_surface[RINDEX(gridp,ray)] * exp(-tau) / 2.0;
    }
  }


  return k;
}

/*-----------------------------------------------------------------------------------------------*/
