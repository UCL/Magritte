// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <stdio.h>
#include <string.h>
#include <math.h>
#include <iostream>

#include "declarations.hpp"
#include "calc_reac_rates.hpp"
#include "species_tools.hpp"



// rate_H2_formation: returns rate coefficient for H2 formation reaction
// ---------------------------------------------------------------------

double rate_H2_formation (CELL *cell, REACTION *reaction, int reac, long o)
{

  // Copy reaction data to variables with more convenient names

  double alpha = reaction[reac].alpha;
  double beta  = reaction[reac].beta;
  double gamma = reaction[reac].gamma;

  double RT_min = reaction[reac].RT_min;
  double RT_max = reaction[reac].RT_max;


  // Following Cazaux & Tielens (2002, ApJ, 575, L29) and (2004, ApJ, 604, 222)


  // Mean thermal speed of hydrogen atoms (cm s^-1)

  double thermal_speed = 1.45E5 * sqrt(cell[o].temperature.gas / 1.0E2);


  // Thermally averaged sticking coefficient of H atoms on grains
  // following Hollenbach & McKee (1979, ApJS, 41, 555, eqn 3.7)

  double sticking_coeff = 1.0 / (1.0 + 0.04*sqrt(cell[o].temperature.gas + cell[o].temperature.dust)
                                 + 0.2*cell[o].temperature.gas/100.0 + 0.08*pow(cell[o].temperature.gas/100.0,2));


  // Flux of H atoms in monolayers per second (mLy s^-1)

  double flux = 1.0E-10;


  // Cross sections

  double cs_tot = 6.273E-22;   // Total mixed grain cross section per H nucleus (cm^-2/nucleus)
  double cs_sil = 8.473E-22;   // Silicate grain cross section per H nucleus (cm^-2/nucleus)
  double cs_gra = 7.908E-22;   // Graphite grain cross section per H nucleus (cm^-2/nucleus)


  // Silicate grain properties (Table 1 in Cazeaux & Tielens, 2002)

  double mu_sil    = 0.005;   // Fraction of newly formed H2 that stays on grain surface
  double E_s_sil   = 110.0;   // Physi- chemisorbed saddle point energy (K)
  double E_H2_sil  = 320.0;   // Desorption energy of H2 molecules (K)
  double E_Hph_sil = 450.0;   // Desorption energy of physisorbed H atoms (K)
  double E_Hch_sil = 3.0E4;   // Desorption energy of chemisorbed H atoms (K)
  double nu_H2_sil = 3.0E12;  // Vibrational frequency of H2 in their surface sites (s^-1)
  double nu_H_sil  = 1.3E13;  // Vibrational frequency of H  in their surface sites (s^-1)


  // Calculate formation efficiency on silicate grains

  double factor1_sil = mu_sil*flux / (2.0*nu_H2_sil*exp(-E_H2_sil/cell[o].temperature.dust));

  double factor2_sil = pow(1.0 + sqrt( (E_Hch_sil-E_s_sil) / (E_Hph_sil-E_s_sil) ), 2)
                       / 4.0 * exp(-E_s_sil/cell[o].temperature.dust);

  double xi_sil = 1.0 / (1.0 + nu_H_sil / (2.0*flux) * exp(-1.5*E_Hch_sil/cell[o].temperature.dust)
                               * pow(1.0 + sqrt( (E_Hch_sil-E_s_sil) / (E_Hph_sil-E_s_sil) ), 2));


  double formation_efficiency_sil = 1.0 / (1.0 + factor1_sil + factor2_sil) * xi_sil;


  // Graphite grain properties (Table 2 in Cazeaux & Tielens, 2004)

  double mu_gra    = 0.005;    // Fraction of newly formed H2 that stays on grain surface
  double E_s_gra   = 260.0;    // Physi- chemisorbed saddle point energy (K)
  double E_H2_gra  = 520.0;    // Desorption energy of H2 molecules (K)
  double E_Hph_gra = 800.0;    // Desorption energy of physisorbed H atoms (K)
  double E_Hch_gra = 3.0E4;    // Desorption energy of chemisorbed H atoms (K)
  double nu_H2_gra = 3.0E12;   // Vibrational frequency of H2 in their surface sites (s^-1)
  double nu_H_gra  = 1.3E13;   // Vibrational frequency of H  in their surface sites (s^-1)


  // Calculate formation efficiency on graphite grains

  double factor1_gra = mu_gra*flux / (2.0*nu_H2_gra*exp(-E_H2_gra/cell[o].temperature.dust));

  double factor2_gra = pow(1.0 + sqrt( (E_Hch_gra-E_s_gra) / (E_Hph_gra-E_s_gra) ), 2)
                       / 4.0 * exp(-E_s_gra/cell[o].temperature.dust);


  double xi_gra = 1.0 / (1.0 + nu_H_gra / (2.0*flux) * exp(-1.5*E_Hch_gra/cell[o].temperature.dust)
                               * pow(1.0 + sqrt( (E_Hch_gra-E_s_gra) / (E_Hph_gra-E_s_gra) ), 2));


  double formation_efficiency_gra = 1.0 / (1.0 + factor1_gra + factor2_gra) * xi_gra;


  // Calculate reaction coefficient by combining formation on silicate and graphite

  return 0.5 * thermal_speed * sticking_coeff * METALLICITY * 100.0 / GAS_TO_DUST
             * (cs_sil*formation_efficiency_sil + cs_gra*formation_efficiency_gra);

}




// rate_PAH: returns rate coefficient for reactions with PAHs
// ----------------------------------------------------------

double rate_PAH (CELL *cell, REACTION *reaction, int reac, long o)
{

  // Copy reaction data to variables with more convenient names

  double alpha = reaction[reac].alpha;
  double beta  = reaction[reac].beta;
  double gamma = reaction[reac].gamma;

  double RT_min = reaction[reac].RT_min;
  double RT_max = reaction[reac].RT_max;


  // Following  Wolfire et al. (2003, ApJ, 587, 278; 2008, ApJ, 680, 384)


  // PAH reaction rate parameter, see (Wolfire et al., 2008)

  double phi_PAH = 0.4;

  return alpha * pow(cell[o].temperature.gas/100.0, beta) * phi_PAH;

}




// rate_CRP: returns rate coefficient for reaction induced by cosmic rays
// ----------------------------------------------------------------------

double rate_CRP (CELL *cell, REACTION *reaction, int reac, long o)
{

  // Copy reaction data to variables with more convenient names

  double alpha = reaction[reac].alpha;
  double beta  = reaction[reac].beta;
  double gamma = reaction[reac].gamma;

  double RT_min = reaction[reac].RT_min;
  double RT_max = reaction[reac].RT_max;


  /* Check for large negative gamma values that might cause discrepant
     rates at low temperatures. Set these rates to zero when T < RTMIN. */

  if ( (gamma < -200.0) && (cell[o].temperature.gas < RT_min) )
  {
    return 0.0;
  }

  else if ( ( (cell[o].temperature.gas <= RT_max) || (RT_max == 0.0) )
            && no_better_data(reac, reaction, cell[o].temperature.gas) )
  {
    return alpha * ZETA;
  }


  return 0.0;

}




// rate_CRPHOT: returns rate coefficient for reaction induced by cosmic rays
// -------------------------------------------------------------------------

double rate_CRPHOT (CELL *cell, REACTION *reaction, int reac, long o)
{

  // Copy reaction data to variables with more convenient names

  double alpha = reaction[reac].alpha;
  double beta  = reaction[reac].beta;
  double gamma = reaction[reac].gamma;

  double RT_min = reaction[reac].RT_min;
  double RT_max = reaction[reac].RT_max;


  /* Check for large negative gamma values that might cause discrepant
     rates at low temperatures. Set these rates to zero when T < RTMIN. */

  if ( (gamma < -200.0) && (cell[o].temperature.gas < RT_min) )
  {
    return 0.0;
  }

  else if ( ( (cell[o].temperature.gas <= RT_max) || (RT_max == 0.0) )
            && no_better_data(reac, reaction, cell[o].temperature.gas) )
  {
    return alpha * ZETA * pow(cell[o].temperature.gas/300.0, beta) * gamma / (1.0 - OMEGA);
  }


  return 0.0;

}




// rate_FREEZE: returns rate coefficient for freeze-out reaction of neutral species
// --------------------------------------------------------------------------------

double rate_FREEZE (CELL *cell, REACTION *reaction, int reac, long o)
{

  // Copy reaction data to variables with more convenient names

  double alpha = reaction[reac].alpha;
  double beta  = reaction[reac].beta;
  double gamma = reaction[reac].gamma;

  double RT_min = reaction[reac].RT_min;
  double RT_max = reaction[reac].RT_max;


  double sticking_coeff = 0.3;      // dust grain sticking coefficient
  double grain_param    = 2.4E-22;  // <d_g a^2> average grain density times radius squared (cm^2)
                                    // = average grain surface area per H atom (devided by PI)

  double radius_grain = 1.0E-7;     // radius of dust grains
  // double radius_grain = 1.0E-5;     // radius of dust grains


  double C_ion = 0.0;               // Factor taking care of electrostatic effects


  // Following Roberts et al. 2007
  // equation (6)

  if      (beta == 0.0)   // For neutral species
  {
    C_ion = 1.0;
  }

  else if (beta == 1.0)   // For (singly) charged species
  {
    C_ion = 1.0 + 16.71E-4/(radius_grain*cell[o].temperature.gas);
  }

  else
  {
    C_ion = 0.0;
  }

  // Rawlings et al. 1992
  // Roberts et al. 2007, equation (5)

  return alpha * 4.57E4 * grain_param * sqrt(cell[o].temperature.gas/gamma) * C_ion * sticking_coeff;

}




// rate_ELFRZE: returns rate coefficient for freeze-out reaction of singly charged positive ions
// ---------------------------------------------------------------------------------------------

double rate_ELFRZE (CELL *cell, REACTION *reaction, int reac, long o)
{

  // Copy reaction data to variables with more convenient names

  double alpha = reaction[reac].alpha;
  double beta  = reaction[reac].beta;
  double gamma = reaction[reac].gamma;

  double RT_min = reaction[reac].RT_min;
  double RT_max = reaction[reac].RT_max;


  double sticking_coeff = 0.3;       // dust grain sticking coefficient
  double grain_param    = 2.4E-22;   // <d_g a^2> average grain density times radius squared (cm^2)
                                     // = average grain surface area per H atom (devided by PI)
  double radius_grain = 1.0E-7;      // radius of the dust grains
  // double radius_grain = 1.0E-5;     // radius of the dust grains


  double C_ion = 0.0;   // Factor taking care of electrostatic effects


  // Following Roberts et al. 2007
  // equation (6)

  if      (beta == 0.0)
  {
    C_ion = 1.0;
  }

  else if (beta == 1.0 )
  {
    C_ion = 1.0 + 16.71E-4/(radius_grain*cell[o].temperature.gas);
  }

  else
  {
    C_ion = 0.0;
  }

  return alpha * 4.57E4 * grain_param * sqrt(cell[o].temperature.gas/gamma) * C_ion * sticking_coeff;

}




// rate_CRH: returns rate coefficient for desorption due to cosmic ray heating
// ---------------------------------------------------------------------------

double rate_CRH (REACTION *reaction, int reac)
{

  // Copy reaction data to variables with more convenient names

  double alpha = reaction[reac].alpha;
  double beta  = reaction[reac].beta;
  double gamma = reaction[reac].gamma;

  double RT_min = reaction[reac].RT_min;
  double RT_max = reaction[reac].RT_max;


  double yield       = 0.0;       // Number of adsorbed molecules released per cosmic ray impact
  double flux        = 2.06E-3;   // Flux of iron nuclei cosmic rays (in cm^-2 s^-1)
  double grain_param = 2.4E-22;   // <d_g a^2> average grain density times radius squared (cm^2)


  // Following Roberts et al. (2007, MNRAS, 382, 773, Equation 3)

  if (gamma < 1210.0)
  {
    yield = 1.0E5;
  }

  else
  {
    yield = 0.0;
  }

  return flux * ZETA * grain_param * yield;

}




// rate_THERM: returns rate coefficient for thermal desorption
// -----------------------------------------------------------

double rate_THERM (CELL *cell, REACTION *reaction, int reac, long o)
{

  // Copy reaction data to variables with more convenient names

  double alpha = reaction[reac].alpha;
  double beta  = reaction[reac].beta;
  double gamma = reaction[reac].gamma;

  double RT_min = reaction[reac].RT_min;
  double RT_max = reaction[reac].RT_max;


  // Following Hasegawa, Herbst & Leung (1992, ApJS, 82, 167, Equations 2 & 3)

  return sqrt(2.0 * 1.5E15 * KB / (PI*PI*AU) * alpha / gamma) * exp(-alpha/cell[o].temperature.dust);

}




// rate_GM: returns rate coefficient for grain mantle reactions
// ------------------------------------------------------------

double rate_GM (REACTION *reaction, int reac)
{

  // Copy reaction data to variables with more convenient names

  double alpha = reaction[reac].alpha;
  double beta  = reaction[reac].beta;
  double gamma = reaction[reac].gamma;

  double RT_min = reaction[reac].RT_min;
  double RT_max = reaction[reac].RT_max;


  return alpha;

}




// rate_canonical: returns canonical rate coefficient for reaction
// ---------------------------------------------------------------

double rate_canonical (CELL *cell, REACTION *reaction, int reac, long o)
{

  // Copy reaction data to variables with more convenient names

  double alpha  = reaction[reac].alpha;
  double beta   = reaction[reac].beta;
  double gamma  = reaction[reac].gamma;
  double RT_min = reaction[reac].RT_min;
  double RT_max = reaction[reac].RT_max;


  /* Check for large negative gamma values that might cause discrepant
     rates at low temperatures. Set these rates to zero when T < RTMIN. */

  if ( (gamma < -200.0) && (cell[o].temperature.gas < RT_min) )
  {
    return 0.0;
  }

  else if ( ( (cell[o].temperature.gas <= RT_max) || (RT_max == 0.0) )
            && no_better_data(reac, reaction, cell[o].temperature.gas) )
  {
    return alpha * pow(cell[o].temperature.gas/300.0, beta) * exp(-gamma/cell[o].temperature.gas);
  }


  return 0.0;

}
