/* Frederik De Ceuster - University College London                                               */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/* calc_rate: Calculate reaction rates (the ones not depending on the radiation field)           */
/*                                                                                               */
/* (based on H2_form, shield and calc_reac_rates in 3D-PDR)    v                                 */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/

#include <stdio.h>
#include <string.h>
#include <math.h>



/* rate_H2_formation: returns the rate coefficient for the H2 formation reaction                 */
/*-----------------------------------------------------------------------------------------------*/

double rate_H2_formation( REACTIONS *reaction, int reac, double temperature_gas,
                          double temperature_dust, double metallicity, double gas2dust)
{

  double alpha;                             /* alpha coefficient to calculate rate coefficient k */
  double beta;                               /* beta coefficient to calculate rate coefficient k */
  double gamma;                             /* gamma coefficient to calculate rate coefficient k */

  double RT_min;                           /* RT_min coefficient to calculate rate coefficient k */
  double RT_max;                           /* RT_max coefficient to calculate rate coefficient k */

  double factor1, factor2;                                                   /* helper variables */

  double xi;                                          /* correction factor for high temperatures */

  double k;                                                              /* reaction coefficient */



  /* Copy the reaction data to variables with more convenient names */

  alpha = reaction[reac].alpha;
  beta = reaction[reac].beta;
  gamma = reaction[reac].gamma;

  RT_min = reaction[reac].RT_min;
  RT_max = reaction[reac].RT_max;


  /* Following Cazaux & Tielens (2002, ApJ, 575, L29) and (2004, ApJ, 604, 222) */

  double thermal_speed;                        /* Mean thermal speed of hydrogen atoms (cm s^-1) */

  thermal_speed = 1.45E5 * sqrt(temperature_gas / 1.0E2);


  double sticking_coeff;         /* Thermally averaged sticking coefficient of H atoms on grains */
                                  /* following Hollenbach & McKee (1979, ApJS, 41, 555, eqn 3.7) */

  sticking_coeff = 1.0 / ( 1.0 + 0.04*sqrt(temperature_gas + temperature_dust)
                           + 0.2*temperature_gas/100.0 + 0.08*pow(temperature_gas/100.0,2) );


  double flux = 1.0E-10;                  /* Flux of H atoms in monolayers per second (mLy s^-1) */


  /* Cross sections */

  double cs_tot = 6.273E-22;    /* Total mixed grain cross section per H nucleus (cm^-2/nucleus) */
  double cs_sil = 8.473E-22;       /* Silicate grain cross section per H nucleus (cm^-2/nucleus) */
  double cs_gra = 7.908E-22;       /* Graphite grain cross section per H nucleus (cm^-2/nucleus) */


  /* Silicate grain properties (Table 1 in Cazeaux & Tielens, 2002) */

  double mu_sil    = 0.005;       /* Fraction of newly formed H2 that stays on the grain surface */
  double E_s_sil   = 110.0;                        /* Physi- chemisorbed saddle point energy (K) */
  double E_H2_sil  = 320.0;                             /* Desorption energy of H2 molecules (K) */
  double E_Hph_sil = 450.0;                      /* Desorption energy of physisorbed H atoms (K) */
  double E_Hch_sil = 3.0E4;                      /* Desorption energy of chemisorbed H atoms (K) */
  double nu_H2_sil = 3.0E12;        /* Vibrational frequency of H2 in their surface sites (s^-1) */
  double nu_H_sil  = 1.3E13;         /* Vibrational frequency of H in their surface sites (s^-1) */


  /* Calculate the formation efficiency on silicate grains */

  factor1 = mu_sil*flux / ( 2.0*nu_H2_sil*exp(-E_H2_sil/temperature_dust) );

  factor2 = pow(1.0 + sqrt( (E_Hch_sil-E_s_sil) / (E_Hph_sil-E_s_sil) ), 2)
            / 4.0 * exp(-E_s_sil/temperature_dust);

  xi = 1.0 / ( 1.0 + nu_H_sil / (2.0*flux) * exp(-1.5*E_Hch_sil/temperature_dust)
                     * pow(1.0 + sqrt( (E_Hch_sil-E_s_sil) / (E_Hph_sil-E_s_sil) ), 2) );


  double formation_efficiency_sil;             /* H2 formation efficiency on silicate grains */

  formation_efficiency_sil = 1.0 / ( 1.0 + factor1 + factor2 ) * xi;


  /* Graphite grain properties (Table 2 in Cazeaux & Tielens, 2004) */

  double mu_gra     = 0.005;      /* Fraction of newly formed H2 that stays on the grain surface */
  double E_s_gra    = 260.0;                       /* Physi- chemisorbed saddle point energy (K) */
  double E_H2_gra   = 520.0;                            /* Desorption energy of H2 molecules (K) */
  double E_Hph_gra  = 800.0;                     /* Desorption energy of physisorbed H atoms (K) */
  double E_Hch_gra  = 3.0E4;                     /* Desorption energy of chemisorbed H atoms (K) */
  double nu_H2_gra = 3.0E12;        /* Vibrational frequency of H2 in their surface sites (s^-1) */
  double nu_H_gra   = 1.3E13;        /* Vibrational frequency of H in their surface sites (s^-1) */

  factor1 = mu_gra*flux / ( 2.0*nu_H2_gra*exp(-E_H2_gra/temperature_dust) );

  factor2 = pow(1.0 + sqrt( (E_Hch_gra-E_s_gra) / (E_Hph_gra-E_s_gra) ), 2)
            / 4.0 * exp(-E_s_gra/temperature_dust);

  xi = 1.0 / ( 1.0 + nu_H_gra / (2.0*flux) * exp(-1.5*E_Hch_gra/temperature_dust)
                     * pow(1.0 + sqrt( (E_Hch_gra-E_s_gra) / (E_Hph_gra-E_s_gra) ), 2) );

  double formation_efficiency_gra;                 /* H2 formation efficiency on graphite grains */
  formation_efficiency_gra = 1.0 / ( 1.0 + factor1 + factor2 ) * xi;


  /* Calculate reaction coefficient */
  /* Combine the formation on silicate and graphite */

  return k = 0.5 * thermal_speed
             * (cs_sil*formation_efficiency_sil + cs_gra*formation_efficiency_gra)
             * sticking_coeff * metallicity * 100.0 / gas2dust;
}

/*-----------------------------------------------------------------------------------------------*/





/* rate_PAH: returns the rate coefficient for the reactions with PAHs                            */
/*-----------------------------------------------------------------------------------------------*/

double rate_PAH( REACTIONS *reaction, int reac, double temperature_gas)
{

  double alpha;                             /* alpha coefficient to calculate rate coefficient k */
  double beta;                               /* beta coefficient to calculate rate coefficient k */
  double gamma;                             /* gamma coefficient to calculate rate coefficient k */

  double RT_min;                           /* RT_min coefficient to calculate rate coefficient k */
  double RT_max;                           /* RT_max coefficient to calculate rate coefficient k */

  int rc;                                                                      /* reaction index */

  double k;                                                              /* reaction coefficient */



  /* Copy the reaction data to variables with more convenient names */

  alpha = reaction[reac].alpha;
  beta = reaction[reac].beta;
  gamma = reaction[reac].gamma;

  RT_min = reaction[reac].RT_min;
  RT_max = reaction[reac].RT_max;


  /* Following  Wolfire et al. (2003, ApJ, 587, 278; 2008, ApJ, 680, 384) */

  double phi_PAH = 0.4;               /* PAH reaction rate parameter, see (Wolfire et al., 2008) */

  return k = alpha * pow(temperature_gas/100.0, beta) * phi_PAH;
}

/*-----------------------------------------------------------------------------------------------*/





/* rate_CRP: returns the rate coefficient for the reaction induced by cosmic rays                */
/*-----------------------------------------------------------------------------------------------*/

double rate_CRP( REACTIONS *reaction, int reac, double temperature_gas)
{

  double alpha;                             /* alpha coefficient to calculate rate coefficient k */
  double beta;                               /* beta coefficient to calculate rate coefficient k */
  double gamma;                             /* gamma coefficient to calculate rate coefficient k */

  double RT_min;                           /* RT_min coefficient to calculate rate coefficient k */
  double RT_max;                           /* RT_max coefficient to calculate rate coefficient k */

  int rc;                                                                      /* reaction index */

  double k;                                                              /* reaction coefficient */

  double zeta = 1.0;                                                      /* cosmic ray variable */


  /* For all duplicates */
  /* duplicates is 1 if there is only on entry for the reaction (different from 3D-PDR) */

  for (rc=0; rc<reaction[reac].dup; rc++){


    /* Copy the reaction data to variables with more convenient (shorter) names */

    alpha = reaction[reac+rc].alpha;
    beta  = reaction[reac+rc].beta;
    gamma = reaction[reac+rc].gamma;

    RT_min = reaction[reac+rc].RT_min;
    RT_max = reaction[reac+rc].RT_max;


    /* Check for large negative gamma values that might cause discrepant
       rates at low temperatures. Set these rates to zero when T < RTMIN. */

    if ( gamma < -200.0  &&  temperature_gas < RT_min ){

      return k = 0.0;
    }

    else if ( temperature_gas < RT_max ){

      return k = alpha * zeta;
    }

  } /* end of rc loop over duplicates */

  return k = 0.0;
}

/*-----------------------------------------------------------------------------------------------*/





/* rate_CRPHOT: returns the rate coefficient for the reaction induced by cosmic rays             */
/*-----------------------------------------------------------------------------------------------*/

double rate_CRPHOT( REACTIONS *reaction, int reac, double temperature_gas)
{

  double alpha;                             /* alpha coefficient to calculate rate coefficient k */
  double beta;                               /* beta coefficient to calculate rate coefficient k */
  double gamma;                             /* gamma coefficient to calculate rate coefficient k */

  double RT_min;                           /* RT_min coefficient to calculate rate coefficient k */
  double RT_max;                           /* RT_max coefficient to calculate rate coefficient k */

  int rc;                                                                      /* reaction index */

  double k;                                                              /* reaction coefficient */

  double zeta = 1.0;                                                      /* cosmic ray variable */
  double omega = 0.42;                                 /* variable for second cosmic ray photons */


  /* For all duplicates */
  /* duplicates is 1 if there is only on entry for the reaction (different from 3D-PDR) */

  for (rc=0; rc<reaction[reac].dup; rc++){


    /* Copy the reaction data to variables with more convenient (shorter) names */

    alpha = reaction[reac+rc].alpha;
    beta  = reaction[reac+rc].beta;
    gamma = reaction[reac+rc].gamma;

    RT_min = reaction[reac+rc].RT_min;
    RT_max = reaction[reac+rc].RT_max;


    /* Check for large negative gamma values that might cause discrepant
       rates at low temperatures. Set these rates to zero when T < RTMIN. */

    if ( gamma < -200.0  &&  temperature_gas < RT_min ){

      return k = 0.0;
    }

    else if ( temperature_gas < RT_max ){

      return k = alpha * zeta * pow(temperature_gas/300.0, beta) * gamma / (1.0 - omega);
    }

  } /* end of rc loop over duplicates */

  return k = 0.0;
}

/*-----------------------------------------------------------------------------------------------*/





/* rate_FREEZE: returns the rate coefficient for freeze-out reaction of neutral species          */
/*-----------------------------------------------------------------------------------------------*/

double rate_FREEZE( REACTIONS *reaction, int reac, double temperature_gas)
{

  double alpha;                             /* alpha coefficient to calculate rate coefficient k */
  double beta;                               /* beta coefficient to calculate rate coefficient k */
  double gamma;                             /* gamma coefficient to calculate rate coefficient k */

  double RT_min;                           /* RT_min coefficient to calculate rate coefficient k */
  double RT_max;                           /* RT_max coefficient to calculate rate coefficient k */

  int rc;                                                                      /* reaction index */

  double sticking_coeff = 0.3;                                /* dust grain sticking coefficient */
  double grain_param = 2.4E-22;   /* <d_g a^2> average grain density times radius squared (cm^2) */
                                      /* = average grain surface area per H atom (devided by PI) */
  double grain_radius = 1.0E-5;                                     /* radius of the dust grains */
  // double grain_radius = 1.0E-7;                                     /* radius of the dust grains */


  double C_ion;                                   /* Factor taking care of electrostatic effects */

  double k;                                                              /* reaction coefficient */


  alpha = reaction[reac].alpha;
  beta  = reaction[reac].beta;
  gamma = reaction[reac].gamma;

  RT_min = reaction[reac].RT_min;
  RT_max = reaction[reac].RT_max;


  /* Following Roberts et al. 2007 */
  /* equation (6) */

  if ( beta == 0.0 ){


    /* For neutral species */

    C_ion = 1.0;
  }

  else if (beta == 1.0 ){


    /* For (singly) charged species */

    C_ion = 1.0 + 16.71E-4/(grain_radius*temperature_gas);
  }

  else {

    C_ion = 0.0;
  }

  /* Rawlings et al. 1992 */
  /* Roberts et al. 2007, equation (5) */

  return k = alpha * 4.57E4 * grain_param * sqrt(temperature_gas/gamma) * C_ion * sticking_coeff;

}

/*-----------------------------------------------------------------------------------------------*/





/* rate_ELFRZE: returns rate coefficient for freeze-out reaction of singly charged positive ions */
/*-----------------------------------------------------------------------------------------------*/

double rate_ELFRZE( REACTIONS *reaction, int reac, double temperature_gas)
{

  double alpha;                             /* alpha coefficient to calculate rate coefficient k */
  double beta;                               /* beta coefficient to calculate rate coefficient k */
  double gamma;                             /* gamma coefficient to calculate rate coefficient k */

  double RT_min;                           /* RT_min coefficient to calculate rate coefficient k */
  double RT_max;                           /* RT_max coefficient to calculate rate coefficient k */

  int rc;                                                                      /* reaction index */

  double sticking_coeff = 0.3;                                /* dust grain sticking coefficient */
  double grain_param = 2.4E-22;   /* <d_g a^2> average grain density times radius squared (cm^2) */
                                      /* = average grain surface area per H atom (devided by PI) */
  double grain_radius = 1.0E-5;                                     /* radius of the dust grains */
  // double grain_radius = 1.0E-7;                                     /* radius of the dust grains */


  double C_ion;                                   /* Factor taking care of electrostatic effects */

  double k;                                                              /* reaction coefficient */


  alpha = reaction[reac].alpha;
  beta  = reaction[reac].beta;
  gamma = reaction[reac].gamma;

  RT_min = reaction[reac].RT_min;
  RT_max = reaction[reac].RT_max;


  /* Following Roberts et al. 2007 */
  /* equation (6) */

  if ( beta == 0.0 ){

    C_ion = 1.0;
  }

  else if (beta == 1.0 ){

    C_ion = 1.0 + 16.71E-4/(grain_radius*temperature_gas);
  }

  else {

    C_ion = 0.0;
  }

  return k = alpha * 4.57E4 * grain_param * sqrt(temperature_gas/gamma) * C_ion * sticking_coeff;

}

/*-----------------------------------------------------------------------------------------------*/





/* rate_CRH: returns rate coefficient for desorption due to cosmic ray heating                   */
/*-----------------------------------------------------------------------------------------------*/

double rate_CRH( REACTIONS *reaction, int reac, double temperature_gas)
{

  double alpha;                             /* alpha coefficient to calculate rate coefficient k */
  double beta;                               /* beta coefficient to calculate rate coefficient k */
  double gamma;                             /* gamma coefficient to calculate rate coefficient k */

  double RT_min;                           /* RT_min coefficient to calculate rate coefficient k */
  double RT_max;                           /* RT_max coefficient to calculate rate coefficient k */

  int rc;                                                                      /* reaction index */

  double zeta;                                                           /* Cosmic ray parameter */

  double yield;                   /* Number of adsorbed molecules released per cosmic ray impact */
  double flux = 2.06E-3;                      /* Flux of iron nuclei cosmic rays (in cm^-2 s^-1) */
  double grain_param = 2.4E-22;   /* <d_g a^2> average grain density times radius squared (cm^2) */

  double k;                                                              /* reaction coefficient */


  alpha = reaction[reac].alpha;
  beta  = reaction[reac].beta;
  gamma = reaction[reac].gamma;

  RT_min = reaction[reac].RT_min;
  RT_max = reaction[reac].RT_max;


  /* Following Roberts et al. (2007, MNRAS, 382, 773, Equation 3) */

  if ( gamma < 1210.0 ){

    yield = 1.0E5;
  }

  else {

    yield = 0.0;
  }

  return k = flux * zeta * grain_param * yield;
}

/*-----------------------------------------------------------------------------------------------*/





/* rate_THERM: returns rate coefficient for thermal desorption                                   */
/*-----------------------------------------------------------------------------------------------*/

double rate_THERM( REACTIONS *reaction, int reac, double temperature_gas, double temperature_dust)
{

  double alpha;                             /* alpha coefficient to calculate rate coefficient k */
  double beta;                               /* beta coefficient to calculate rate coefficient k */
  double gamma;                             /* gamma coefficient to calculate rate coefficient k */

  double RT_min;                           /* RT_min coefficient to calculate rate coefficient k */
  double RT_max;                           /* RT_max coefficient to calculate rate coefficient k */

  int rc;                                                                      /* reaction index */

  double zeta;                                                           /* Cosmic ray parameter */

  double yield;                   /* Number of adsorbed molecules released per cosmic ray impact */
  double flux;                                /* Flux of iron nuclei cosmic rays (in cm^-2 s^-1) */

  double k;                                                              /* reaction coefficient */


  alpha = reaction[reac].alpha;
  beta  = reaction[reac].beta;
  gamma = reaction[reac].gamma;

  RT_min = reaction[reac].RT_min;
  RT_max = reaction[reac].RT_max;


  /* Following Hasegawa, Herbst & Leung (1992, ApJS, 82, 167, Equations 2 & 3) */

  return k = sqrt(2.0 * 1.5E15 * KB / (PI*PI*AU) * alpha / gamma) * exp(-alpha/temperature_dust);
}

/*-----------------------------------------------------------------------------------------------*/





/* rate_GM: returns rate coefficient for grain mantle reactions                                  */
/*-----------------------------------------------------------------------------------------------*/

double rate_GM( REACTIONS *reaction, int reac )
{

  double alpha;                             /* alpha coefficient to calculate rate coefficient k */
  double beta;                               /* beta coefficient to calculate rate coefficient k */
  double gamma;                             /* gamma coefficient to calculate rate coefficient k */

  double RT_min;                           /* RT_min coefficient to calculate rate coefficient k */
  double RT_max;                           /* RT_max coefficient to calculate rate coefficient k */

  int rc;                                                                      /* reaction index */

  double zeta;                                                           /* Cosmic ray parameter */

  double yield;                   /* Number of adsorbed molecules released per cosmic ray impact */
  double flux;                                /* Flux of iron nuclei cosmic rays (in cm^-2 s^-1) */

  double k;                                                              /* reaction coefficient */


  alpha = reaction[reac].alpha;
  beta  = reaction[reac].beta;
  gamma = reaction[reac].gamma;

  RT_min = reaction[reac].RT_min;
  RT_max = reaction[reac].RT_max;


  return k = alpha;
}

/*-----------------------------------------------------------------------------------------------*/





/* rate_canonical: returns the canonical rate coefficient for the reaction                       */
/*-----------------------------------------------------------------------------------------------*/

double rate_canonical( REACTIONS *reaction, int reac, double temperature_gas)
{

  double alpha;                             /* alpha coefficient to calculate rate coefficient k */
  double beta;                               /* beta coefficient to calculate rate coefficient k */
  double gamma;                             /* gamma coefficient to calculate rate coefficient k */

  double RT_min;                           /* RT_min coefficient to calculate rate coefficient k */
  double RT_max;                           /* RT_max coefficient to calculate rate coefficient k */

  int rc;                                                                      /* reaction index */

  double k;                                                              /* reaction coefficient */



  /* For all duplicates */
  /* duplicates is 1 if there is only on entry for the reaction (different from 3D-PDR) */

  for (rc=0; rc<reaction[reac].dup; rc++){


    /* Copy the reaction data to variables with more convenient (shorter) names */

    alpha = reaction[reac+rc].alpha;
    beta  = reaction[reac+rc].beta;
    gamma = reaction[reac+rc].gamma;

    RT_min = reaction[reac+rc].RT_min;
    RT_max = reaction[reac+rc].RT_max;


    /* Check for large negative gamma values that might cause discrepant
       rates at low temperatures. Set these rates to zero when T < RTMIN. */

    if ( gamma < -200.0  &&  temperature_gas < RT_min ){

      return k = 0.0;
    }

    else if ( temperature_gas < RT_max ){

      return k = alpha * pow(temperature_gas/300.0, beta) * exp(-gamma/temperature_gas);
    }

  } /* end of rc loop over duplicates */

  return k = 0.0;
}

/*-----------------------------------------------------------------------------------------------*/
