// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>

#include "declarations.hpp"
#include "calc_reac_rates_rad.hpp"
#include "radfield_tools.hpp"


/* Note in the arguments that the temperatures are local (doubles), but rad_surface, AV and column
   densities are still represented by the pointers to the full arrays */


// rate_PHOTD: returns rate coefficient for photodesorption
// --------------------------------------------------------

double rate_PHOTD (CELLS *cells, REACTIONS reactions, int e, long o)
{

  // Copy reaction data to variables with more convenient names

  double alpha  = reactions.alpha[e];
  double beta   = reactions.beta[e];
  double gamma  = reactions.gamma[e];
  double RT_min = reactions.RT_min[e];
  double RT_max = reactions.RT_max[e];


  double yield;   // Number of adsorbed molecules released per cosmic ray impact

  double flux = 1.7E8;   // Flux of FUV photons in unattenuated Draine field ( photons cm^-2 s^-1)
  // double flux = 1.0E8;   // Flux of FUV photons in unattenuated Habing field ( photons cm^-2 s^-1)

  double grain_param = 2.4E-22;   // <d_g a^2> average grain density times radius squared (cm^2)
                                  // = average grain surface area per H atom (devided by PI)


  if      (cells->temperature_gas[o] < 50.0)
  {
    yield = 3.5E-3;
  }

  else if (cells->temperature_gas[o] < 85.0)
  {
    yield = 4.0E-3;
  }

  else if (cells->temperature_gas[o] < 100.0)
  {
    yield = 5.5E-3;
  }

  else
  {
    yield = 7.5E-3;
  }


  double rate = 0.0;   // reactio rate coefficient


  for (long r = 0; r < NRAYS; r++)
  {
    rate = rate + flux * cells->rad_surface[RINDEX(o,r)]
                  * exp(-1.8*cells->AV[RINDEX(o,r)]) * grain_param * yield;
  }


  return rate;

}




// rate_H2_photodissociation: returns rate coefficient for H2 dissociation
// -----------------------------------------------------------------------

double rate_H2_photodissociation (CELLS *cells, REACTIONS reactions, int e, double *column_H2, long o)
{

  // Copy reaction data to variables with more convenient names

  double alpha  = reactions.alpha[e];
  double beta   = reactions.beta[e];
  double gamma  = reactions.gamma[e];
  double RT_min = reactions.RT_min[e];
  double RT_max = reactions.RT_max[e];


  double lambda = 1000.0;                            // wavelength (in Å) of a typical transition

  double doppler_width = V_TURB / (lambda*1.0E-8);   // doppler linewidth (in Hz) of typical transition
                                                     // (assuming turbulent broadening with b = 3 km/s)

  double radiation_width = 8.0E7;                    // radiative linewidth (in Hz) of typical transition


  double rate = 0.0;   // reaction rate coefficient


  for (long r = 0; r < NRAYS; r++)
  {
    rate = rate + alpha * cells->rad_surface[RINDEX(o,r)]
                        * self_shielding_H2 (column_H2[RINDEX(o,r)], doppler_width, radiation_width)
                        * dust_scattering (cells->AV[RINDEX(o,r)], lambda) / 2.0;
  }


  return rate;

}




// rate_CO_photodissociation: returns rate coefficient for CO dissociation
// -----------------------------------------------------------------------

double rate_CO_photodissociation (CELLS *cells, REACTIONS reactions, int e,
                                  double *column_CO, double *column_H2, long o)
{

  // Copy reaction data to variables with more convenient names

  double alpha  = reactions.alpha[e];
  double beta   = reactions.beta[e];
  double gamma  = reactions.gamma[e];
  double RT_min = reactions.RT_min[e];
  double RT_max = reactions.RT_max[e];


  double rate = 0.0;   // reaction rate coefficient


  for (long r = 0; r < NRAYS; r++)
  {

    /* Calculate the mean wavelength (in Å) of the 33 dissociating bands,
       weighted by their fractional contribution to the total shielding
       van Dishoeck & Black (1988, ApJ, 334, 771, Equation 4) */

    double u = log10(1.0 + column_CO[RINDEX(o,r)]);
    double w = log10(1.0 + column_H2[RINDEX(o,r)]);


    /* mean wavelength (in Å) of 33 dissociating bands weighted
       by their fractional contribution to the total shielding */

    double lambda = (5675.0 - 200.6*w) - (571.6 - 24.09*w)*u + (18.22 - 0.7664*w)*u*u;


    /* lambda cannot be larger than the wavelength of band 33 (1076.1Å)
       and cannot be smaller than the wavelength of band 1 (913.6Å) */

    if      (lambda > 1076.1)
    {
      lambda = 1076.1;
    }

    else if (lambda < 913.6)
    {
      lambda = 913.6;
    }


    rate = rate + alpha * cells->rad_surface[RINDEX(o,r)]
                        * self_shielding_CO (column_CO[RINDEX(o,r)], column_H2[RINDEX(o,r)])
                        * dust_scattering (cells->AV[RINDEX(o,r)], lambda) / 2.0;
  }


  return rate;

}




// rate_C_photoionization: returns rate coefficient for C photoionization
// ----------------------------------------------------------------------

double rate_C_photoionization (CELLS *cells, REACTIONS reactions, int e,
                               double *column_C, double *column_H2, long o)
{

  // Copy reaction data to variables with more convenient names

  double alpha  = reactions.alpha[e];
  double beta   = reactions.beta[e];
  double gamma  = reactions.gamma[e];
  double RT_min = reactions.RT_min[e];
  double RT_max = reactions.RT_max[e];


  double rate = 0.0;   // reaction rate coefficient


  for (long r = 0; r < NRAYS; r++)
  {

    /* Calculate the optical depth in the C absorption band, accounting
       for grain extinction and shielding by C and overlapping H2 lines */

    double tau_C = gamma*cells->AV[RINDEX(o,r)] + 1.1E-17*column_C[RINDEX(o,r)]
                   + ( 0.9*pow(cells->temperature_gas[o],0.27)
                          * pow(column_H2[RINDEX(o,r)]/1.59E21, 0.45) );


    // Calculate the C photoionization rate

    rate = rate + alpha * cells->rad_surface[RINDEX(o,r)] * exp(-tau_C) / 2.0;
  }


  return rate;

}




// rate_SI_photoionization: returns rate coefficient for SI photoionization
// ------------------------------------------------------------------------

double rate_SI_photoionization (CELLS *cells, REACTIONS reactions, int e, long o)
{

  // Copy reaction data to variables with more convenient names

  double alpha  = reactions.alpha[e];
  double beta   = reactions.beta[e];
  double gamma  = reactions.gamma[e];
  double RT_min = reactions.RT_min[e];
  double RT_max = reactions.RT_max[e];


  double rate = 0.0;   // reaction rate coefficient


  for (long r = 0; r < NRAYS; r++)
  {

    /* Calculate the optical depth in the SI absorption band, accounting
       for grain extinction and shielding by ??? */

    double tau_S = gamma*cells->AV[RINDEX(o,r)];


    // Calculate SI photoionization rate

    rate = rate + alpha * cells->rad_surface[RINDEX(o,r)] * exp(-tau_S) / 2.0;
  }


  return rate;

}




// rate_canonical_photoreaction: returns rate coefficient for a canonical photoreaction
// ------------------------------------------------------------------------------------

double rate_canonical_photoreaction (CELLS *cells, REACTIONS reactions, int e, long o)
{

  // Copy reaction data to variables with more convenient names

  double alpha  = reactions.alpha[e];
  double beta   = reactions.beta[e];
  double gamma  = reactions.gamma[e];
  double RT_min = reactions.RT_min[e];
  double RT_max = reactions.RT_max[e];


  double rate = 0.0;   // reaction coefficient


  /* Check for large negative gamma values that might cause discrepant
     rates at low temperatures. Set these rates to zero when T < RTMIN. */

  if ( (gamma < -200.0) && (cells->temperature_gas[o] < RT_min) )
  {
    return rate = 0.0;
  }

  else if ( ( (cells->temperature_gas[o] <= RT_max) || (RT_max == 0.0) )
            && reactions.no_better_data(e, cells->temperature_gas[o]) )
  {

    for (long r = 0; r < NRAYS; r++)
    {
      double tau = gamma*cells->AV[RINDEX(o,r)];

      rate = rate + alpha * cells->rad_surface[RINDEX(o,r)] * exp(-tau) / 2.0;
    }
  }


  return rate;

}
