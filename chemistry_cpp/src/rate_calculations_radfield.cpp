/* Frederik De Ceuster - University College London                                               */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/* rate_calculations_radfield: Calculate rates for reactions depending on the radiation field    */
/*                                                                                               */
/* (based on H2_form, shield and calc_reac_rates in 3D-PDR)                                      */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/

#include <stdio.h>
#include <string.h>
#include <math.h>



/* rate_PHOTD: returns rate coefficient for photodesorption                                      */
/*-----------------------------------------------------------------------------------------------*/

double rate_PHOTD( REACTIONS *reaction, int reac, double temperature_gas,
                   double *rad_surface, double *AV )
{

  long   ray;                                                                       /* ray index */

  double alpha;                             /* alpha coefficient to calculate rate coefficient k */
  double beta;                               /* beta coefficient to calculate rate coefficient k */
  double gamma;                             /* gamma coefficient to calculate rate coefficient k */

  double RT_min;                           /* RT_min coefficient to calculate rate coefficient k */
  double RT_max;                           /* RT_max coefficient to calculate rate coefficient k */

  double yield;                   /* Number of adsorbed molecules released per cosmic ray impact */

  double flux;                                                /* Flux of photons (in cm^-2 s^-1) */

  // flux = 1.0E8; /* Flux of FUV photons in the unattenuated Habing field ( photons cm^-2 s^-1) */

  flux = 1.7E8;    /* Flux of FUV photons in the unattenuated Draine field ( photons cm^-2 s^-1) */

  double grain_param = 2.4E-22;   /* <d_g a^2> average grain density times radius squared (cm^2) */
                                      /* = average grain surface area per H atom (devided by PI) */

  double k;                                                              /* reaction coefficient */


  alpha = reaction[reac].alpha;
  beta  = reaction[reac].beta;
  gamma = reaction[reac].gamma;

  RT_min = reaction[reac].RT_min;
  RT_max = reaction[reac].RT_max;


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

  for (ray=0; ray<NRAYS; ray++){

    k = k + flux * rad_surface[ray] * exp(-1.8*AV[ray]) * grain_param * yield;
  }

  return k;
}

/*-----------------------------------------------------------------------------------------------*/





/* rate_H2_photodissociation: returns rate coefficient for H2 dissociation                       */
/*-----------------------------------------------------------------------------------------------*/

double rate_H2_photodissociation( REACTIONS *reaction, int reac, double *rad_surface,
                                  double *AV, double *column_H2, double v_turb )
{

  long   ray;                                                                       /* ray index */

  double alpha;                             /* alpha coefficient to calculate rate coefficient k */
                             /* in this case the unattenuated photodissociation rate (in cm^3/s) */
  double beta;                               /* beta coefficient to calculate rate coefficient k */
  double gamma;                             /* gamma coefficient to calculate rate coefficient k */

  double RT_min;                           /* RT_min coefficient to calculate rate coefficient k */
  double RT_max;                           /* RT_max coefficient to calculate rate coefficient k */

  double k;                                                              /* reaction coefficient */


  alpha = reaction[reac].alpha;
  beta  = reaction[reac].beta;
  gamma = reaction[reac].gamma;

  RT_min = reaction[reac].RT_min;
  RT_max = reaction[reac].RT_max;

  double lambda = 1000.0;                           /* wavelength (in Å) of a typical transition */

  double doppler_width;                     /* Doppler linewidth (in Hz) of a typical transition */
                                                /* (assuming turbulent broadening with b=3 km/s) */
  doppler_width = v_turb / (lambda*1.0E-8);

  double radiation_width = 8.0E7;         /* radiative linewidth (in Hz) of a typical transition */


  double self_shielding_H2( double column_H2, double doppler_width, double radiation_width );
  double dust_scattering( double AV_ray, double lambda );


  for (ray=0; ray<NRAYS; ray++){

    k = k + alpha * rad_surface[ray] * self_shielding_H2(column_H2[ray], doppler_width, radiation_width)
            * dust_scattering(AV[ray], lambda) / 2.0;
  }

  return k;
}
/*-----------------------------------------------------------------------------------------------*/





/* rate_CO_photodissociation: returns rate coefficient for CO dissociation                       */
/*-----------------------------------------------------------------------------------------------*/

double rate_CO_photodissociation( REACTIONS *reaction, int reac, double *rad_surface,
                                  double *AV, double *column_CO, double *column_H2 )
{

  long   ray;                                                                       /* ray index */

  double alpha;                             /* alpha coefficient to calculate rate coefficient k */
                             /* in this case the unattenuated photodissociation rate (in cm^3/s) */
  double beta;                               /* beta coefficient to calculate rate coefficient k */
  double gamma;                             /* gamma coefficient to calculate rate coefficient k */

  double RT_min;                           /* RT_min coefficient to calculate rate coefficient k */
  double RT_max;                           /* RT_max coefficient to calculate rate coefficient k */

  double lambda;                     /* mean wavelength (in Å) of 33 dissociating bands weighted */
                                      /* by their fractional contribution to the total shielding */

  double k;                                                              /* reaction coefficient */


  alpha = reaction[reac].alpha;
  beta  = reaction[reac].beta;
  gamma = reaction[reac].gamma;

  RT_min = reaction[reac].RT_min;
  RT_max = reaction[reac].RT_max;


  /* Calculate the mean wavelength (in Å) of the 33 dissociating bands,
     weighted by their fractional contribution to the total shielding
     van Dishoeck & Black (1988, ApJ, 334, 771, Equation 4) */

  double u = log10(1.0 + column_CO); /* ??? WHY THE + 1.0 ??? */
  double v = log10(1.0 + column_H2); /* ??? WHY THE + 1.0 ??? */

  lambda = (5675.0 - 200.6*W) - (571.6 - 24.09*W)*U + (18.22 - 0.7664*W)*U*U;


  /* lambda cannot be larger than the wavelength of band 33 (1076.1Å)
     and cannot be smaller than the wavelength of band 1 (913.6Å) */

  if ( lambda > 1076.1 ) {
    lambda = 1076.1;
  }

  if ( lambda < 913.6 ){

    lambda = 913.6;
  }


  double self_shielding_CO( double column_H2, double column_CO);
  double dust_scattering( double AV_ray, double lambda );


  /* Now lambda is calculated */

  for (ray=0; ray<NRAYS; ray++){

    k = k + alpha * rad_surface[ray] * self_shielding_CO(column_CO[ray], column_H2[ray])
            * dust_scattering(AV[ray], lambda) / 2.0;
  }

  return k;
}

/*-----------------------------------------------------------------------------------------------*/





/* rate_CI_photoionization: returns rate coefficient for CI photoionization                      */
/*-----------------------------------------------------------------------------------------------*/

double rate_CI_photoionization( REACTIONS *reaction, int reac, double temperature_gas,
                                double *rad_surface, double *AV,
                                double *column_CI, double *column_H2 )
{

  long   ray;                                                                       /* ray index */

  double alpha;                             /* alpha coefficient to calculate rate coefficient k */
                             /* in this case the unattenuated photodissociation rate (in cm^3/s) */
  double beta;                               /* beta coefficient to calculate rate coefficient k */
  double gamma;                             /* gamma coefficient to calculate rate coefficient k */

  double RT_min;                           /* RT_min coefficient to calculate rate coefficient k */
  double RT_max;                           /* RT_max coefficient to calculate rate coefficient k */

  double tau_C;                                       /* optical depth in the CI absorption band */

  double k;                                                              /* reaction coefficient */


  alpha = reaction[reac].alpha;
  beta  = reaction[reac].beta;
  gamma = reaction[reac].gamma;

  RT_min = reaction[reac].RT_min;
  RT_max = reaction[reac].RT_max;


  for (ray=0; ray<NRAYS; ray++){


    /* Calculate the optical depth in the CI absorption band, accounting
       for grain extinction and shielding by CI and overlapping H2 lines */

    tau_C = gamma*AV[ray]
            + 1.1E-17*column_CI + (0.9 * pow(temperature_gas,0.27) * pow(column_H2/1.59E21, 0.45));


    /* Calculate the CI photoionization rate */

    k = k + alpha * rad_surface[ray] * exp(-tau_C) / 2.0;
  }

  return k;
}

/*-----------------------------------------------------------------------------------------------*/





/* rate_SI_photoionization: returns rate coefficient for SI photoionization                      */
/*-----------------------------------------------------------------------------------------------*/

double rate_SI_photoionization( REACTIONS *reaction, int reac, double *rad_surface, double *AV )
{

  long   ray;                                                                       /* ray index */

  double alpha;                             /* alpha coefficient to calculate rate coefficient k */
                             /* in this case the unattenuated photodissociation rate (in cm^3/s) */
  double beta;                               /* beta coefficient to calculate rate coefficient k */
  double gamma;                             /* gamma coefficient to calculate rate coefficient k */

  double RT_min;                           /* RT_min coefficient to calculate rate coefficient k */
  double RT_max;                           /* RT_max coefficient to calculate rate coefficient k */

  double tau_S;                                       /* optical depth in the SI absorption band */

  double k;                                                              /* reaction coefficient */


  alpha = reaction[reac].alpha;
  beta  = reaction[reac].beta;
  gamma = reaction[reac].gamma;

  RT_min = reaction[reac].RT_min;
  RT_max = reaction[reac].RT_max;


  for (ray=0; ray<NRAYS; ray++){


    /* Calculate the optical depth in the SI absorption band, accounting
       for grain extinction and shielding by ??? */

    tau_S = gamma*AV[ray];


    /* Calculate the SI photoionization rate */

    k = k + alpha * rad_surface[ray] * exp(-tau_S) / 2.0;
  }

  return k;
}

/*-----------------------------------------------------------------------------------------------*/





/* self_shielding_H2: Returns H2 self-shielding function                                         */
/*-----------------------------------------------------------------------------------------------*/

double self_shielding_H2( double column_H2, double doppler_width, double radiation_width )
{


  /* Following Federman, Glassgold & Kwan (1979, ApJ, 227, 466) */

  double J_D;                                          /* Doppler contribution to self-shielding */
  double J_R;                                        /* Radiative contribution to self-shielding */

  double self_shielding;                                        /* total self-shielding function */



  /* Calculate the optical depth at line centre */
  /* = N(H2)*f_para*(πe^2/mc)*f/(√πß) ≈ N(H2)*f_para*(1.5E-2)*f/ß */


  double frac_H2_para = 0.5;                            /* (assume H2_ortho / H2_para ratio = 1) */

  double f_osc = 1.0E-2;                          /* Oscillator strength of a typical transition */

  double tau_D;                                                  /* Optical depth at line centre */

  double PIe2_mc = 1.497358985E-2;         /* PI e^2 / mc, with electron charge (e) and mass (m) */

  /* parameter tau_D (eq. A7) in Federman's paper */

  tau_D = column_H2 * frac_H2_para * PIe2_mc * f_osc / doppler_width;


  /* Calculate the Doppler core contribution to the self-shielding (JD) */
  /* Parameter JD (eq. A8) in Federman's paper */

  if ( tau_D == 0.0 ){

    J_D = 1.0;
  }
  else if ( tau_D < 2.0 ){

    J_D = exp(-0.666666667*tau_D);
  }
  else if ( tau_D < 10.0 ){

    J_D = 0.638 * pow(tau_D, -1.25);
  }
  else if ( tau_D < 100.0 ){

    J_D = 0.505 * pow(tau_D, -1.15);
  }
  else {

    J_D = 0.344 * pow(tau_D, -1.0667);
  }


  /* Calculate the radiative wing contribution to self-shielding (JR) */
  /* Parameter JR (eq. A9) in Federman's paper */

  if (radiation_width == 0.0){

    J_R = 0.0;
  }
  else {

    double sqrt_PI = 1.772453851;                                           /* square root of PI */
    double r  = radiation_width / (sqrt_PI*doppler_width);  /* (equation A2 in Federman's paper) */
    double t1 = 3.02 * pow(R*1.0E3,-0.064);                 /* (equation A6 in Federman's paper) */
    double u1 = sqrt(tau_D*R) / T;                          /* (equation A6 in Federman's paper) */

    J_R = r / ( t1 * sqrt(sqrt_PI/2.0 + u1*u1) );
  }


  /* Calculate the total self-shielding function */

  return self_shielding = JD + JR;


}

/*-----------------------------------------------------------------------------------------------*/





/* self_shielding_CO: Returns CO self-shielding function                                         */
/*-----------------------------------------------------------------------------------------------*/

double self_shielding_CO( double column_H2, double column_CO)
{

  double self_shielding;                                        /* total self-shielding function */




}

/*-----------------------------------------------------------------------------------------------*/





/* dust_scattering: Retuns the attenuation due to scattering by dust                             */
/*-----------------------------------------------------------------------------------------------*/

double dust_scattering( double AV_ray, double lambda )
{

  int i;                                                                                /* index */

  double tau_visual;                         /* optical depth at the visual wavelength (λ=5500Å) */
  double tau_lambda;                                       /* optical depth at wavelength lambda */

  double exponent;                                                            /* helper variable */

  /* Coefficients in equation (1) in Wagenblast & Hartquist 1989                                 */
  /*     A(0)    = a(0)*exp(-k(0)*tau)                                                           */
  /*             = relative intensity decrease for 0 < tau < 1                                   */
  /*     A(I)    = ∑ a(i)*exp(-k(i)*tau) for i=1,5                                               */
  /*               relative intensity decrease for tau ≥ 1                                       */

  double A[] = { 1.000,  2.006, -1.438, 0.7364, -0.5076, -0.0592 };

  /*     K(0)    = see A0                                                                        */
  /*     K(I)    = see A(I)                                                                      */

  double k[] = { 0.7514, 0.8490, 1.013, 1.282,   2.005,   5.832 };

  double dust_scatter;                                  /* attenuation due to scattering by dust */


  /* Calculate the optical depth at visual wavelength */

  tau_visual = AV_ray / 1.086;


  /* Convert the optical depth to that at the desired wavelength */

  tau = tau_visual * XLAMBDA(lambda);


  /* Calculate the attenuation due to scattering by dust */
  /* equation (1) in Wagenblast & Hartquist 1989 */

  if ( tau_lambda < 1.0 ){

    exponent = tau_lambda * k[0];

    if ( exponent < 100.0 ){

      dust_scatter = A[0] * exp(-exponent);
    }
  }

  else {

    for (i=0; i<5; i++){

      exponent = tau_lambda * k[i];

      if ( exponent < 100.0 ){

        dust_scatter = dust_scatter + A[i]*exp(-exponent);
      }
    }
  }

  return dust_scatter

}

/*-----------------------------------------------------------------------------------------------*/
