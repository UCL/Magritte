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
#include <stdlib.h>
#include <math.h>

#include "declarations.hpp"
#include "rate_calculations_radfield.hpp"
#include "spline.hpp"

#define IND(r,c) ((c)+(r)*n)



/* rate_PHOTD: returns rate coefficient for photodesorption                                      */
/*-----------------------------------------------------------------------------------------------*/

double rate_PHOTD( int reac, double temperature_gas, double *rad_surface, double *AV )
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

double rate_H2_photodissociation( int reac, double *rad_surface,
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


  for (ray=0; ray<NRAYS; ray++){

    k = k + alpha * rad_surface[ray] * self_shielding_H2(column_H2[ray], doppler_width, radiation_width)
            * dust_scattering(AV[ray], lambda) / 2.0;
  }

  return k;
}
/*-----------------------------------------------------------------------------------------------*/





/* rate_CO_photodissociation: returns rate coefficient for CO dissociation                       */
/*-----------------------------------------------------------------------------------------------*/

double rate_CO_photodissociation( int reac, double *rad_surface,
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


  /* Now lambda is calculated */

  for (ray=0; ray<NRAYS; ray++){


    /* Calculate the mean wavelength (in Å) of the 33 dissociating bands,
       weighted by their fractional contribution to the total shielding
       van Dishoeck & Black (1988, ApJ, 334, 771, Equation 4) */

    double u = log10(1.0 + column_CO[ray]); /* ??? WHY THE + 1.0 ??? */
    double w = log10(1.0 + column_H2[ray]); /* ??? WHY THE + 1.0 ??? */

    lambda = (5675.0 - 200.6*w) - (571.6 - 24.09*w)*u + (18.22 - 0.7664*w)*u*u;


    /* lambda cannot be larger than the wavelength of band 33 (1076.1Å)
       and cannot be smaller than the wavelength of band 1 (913.6Å) */

    if ( lambda > 1076.1 ) {

      lambda = 1076.1;
    }

    if ( lambda < 913.6 ){

      lambda = 913.6;
    }


    k = k + alpha * rad_surface[ray] * self_shielding_CO(column_CO[ray], column_H2[ray])
            * dust_scattering(AV[ray], lambda) / 2.0;
  }

  return k;
}

/*-----------------------------------------------------------------------------------------------*/





/* rate_C_photoionization: returns rate coefficient for C photoionization                      */
/*-----------------------------------------------------------------------------------------------*/

double rate_C_photoionization( int reac, double temperature_gas,
                               double *rad_surface, double *AV,
                               double *column_C, double *column_H2 )
{

  long   ray;                                                                       /* ray index */

  double alpha;                             /* alpha coefficient to calculate rate coefficient k */
                             /* in this case the unattenuated photodissociation rate (in cm^3/s) */
  double beta;                               /* beta coefficient to calculate rate coefficient k */
  double gamma;                             /* gamma coefficient to calculate rate coefficient k */

  double RT_min;                           /* RT_min coefficient to calculate rate coefficient k */
  double RT_max;                           /* RT_max coefficient to calculate rate coefficient k */

  double tau_C;                                        /* optical depth in the C absorption band */

  double k;                                                              /* reaction coefficient */


  alpha = reaction[reac].alpha;
  beta  = reaction[reac].beta;
  gamma = reaction[reac].gamma;

  RT_min = reaction[reac].RT_min;
  RT_max = reaction[reac].RT_max;


  for (ray=0; ray<NRAYS; ray++){


    /* Calculate the optical depth in the C absorption band, accounting
       for grain extinction and shielding by C and overlapping H2 lines */

    tau_C = gamma*AV[ray] + 1.1E-17*column_C[ray]
            + (0.9 * pow(temperature_gas,0.27) * pow(column_H2[ray]/1.59E21, 0.45));


    /* Calculate the C photoionization rate */

    k = k + alpha * rad_surface[ray] * exp(-tau_C) / 2.0;
  }

  return k;
}

/*-----------------------------------------------------------------------------------------------*/





/* rate_SI_photoionization: returns rate coefficient for SI photoionization                      */
/*-----------------------------------------------------------------------------------------------*/

double rate_SI_photoionization( int reac, double *rad_surface, double *AV )
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
    double t1 = 3.02 * pow(r*1.0E3,-0.064);                 /* (equation A6 in Federman's paper) */
    double u1 = sqrt(tau_D*r) / t1;                          /* (equation A6 in Federman's paper) */

    J_R = r / ( t1 * sqrt(sqrt_PI/2.0 + u1*u1) );
  }


  /* Calculate the total self-shielding function */

  return self_shielding = J_D + J_R;


}

/*-----------------------------------------------------------------------------------------------*/





/* self_shielding_CO: Returns CO self-shielding function                                         */
/*-----------------------------------------------------------------------------------------------*/

double self_shielding_CO( double column_CO, double column_H2 )
{

  /*  12CO line shielding, using the computed values listed in
      van Dishoeck & Black (1988, ApJ, 334, 771, Table 5)

      Appropriate shielding factors are determined by performing a  2-dimensional spline
      interpolation over the values listed in Table 5 of van Dishoeck & Black, which include
      contributions from self-shielding and H2 screening  */


  long i, j;                                                                          /* indices */

  double log10shield;                                           /* total self-shielding function */

  const long m = 8;
  const long n = 6;

  double log10column_CO_grid[m] = {12.0E0, 13.0E0, 14.0E0, 15.0E0, 16.0E0, 17.0E0, 18.0E0, 19.0E0};
  double log10column_H2_grid[n] = {18.0E0, 19.0E0, 20.0E0, 21.0E0, 22.0E0, 23.0E0 };

  double log10shield_CO_grid[n*m]
         = { 0.000E+00, -8.539E-02, -1.451E-01, -4.559E-01, -1.303E+00, -3.883E+00, \
            -1.408E-02, -1.015E-01, -1.612E-01, -4.666E-01, -1.312E+00, -3.888E+00, \
            -1.099E-01, -2.104E-01, -2.708E-01, -5.432E-01, -1.367E+00, -3.936E+00, \
            -4.400E-01, -5.608E-01, -6.273E-01, -8.665E-01, -1.676E+00, -4.197E+00, \
            -1.154E+00, -1.272E+00, -1.355E+00, -1.602E+00, -2.305E+00, -4.739E+00, \
            -1.888E+00, -1.973E+00, -2.057E+00, -2.303E+00, -3.034E+00, -5.165E+00, \
            -2.760E+00, -2.818E+00, -2.902E+00, -3.146E+00, -3.758E+00, -5.441E+00, \
            -4.001E+00, -4.055E+00, -4.122E+00, -4.421E+00, -5.077E+00, -6.446E+00  };

  double *d2log10shield;
  d2log10shield = (double*) malloc( m*n*sizeof(double) );



  /* Write the shield_CO values to a text file (for testing) */

  FILE *sCO = fopen("self_shielding_CO_table.txt", "w");

  if (sCO == NULL){

      printf("Error opening file!\n");
      exit(1);
    }

  for (i=0; i<m; i++){

    for (j=0; j<n; j++){

      fprintf(sCO, "%lE\t%lE\t%lE\n", log10column_CO_grid[i],
                                      log10column_H2_grid[j],
                                      log10shield_CO_grid[IND(i,j)] );
    }
  }

  fclose(sCO);



  /* Calculate the splines for the rows (spline.cpp) */

  splie2( log10column_CO_grid, log10column_H2_grid, log10shield_CO_grid, m, n, d2log10shield );



  /* Scale the variables to get a better spline interpolation */

  double log10column_CO = log10(column_CO+1.0);
  double log10column_H2 = log10(column_H2+1.0);



  /* Enforce the variables to be in the range of the interpolating function */

  if (log10column_CO < log10column_CO_grid[0]){

    log10column_CO = log10column_CO_grid[0];
  }

  if (log10column_H2 < log10column_H2_grid[0]){

    log10column_H2 = log10column_H2_grid[0];
  }

  if (log10column_CO > log10column_CO_grid[m-1]){

    log10column_CO = log10column_CO_grid[m-1];
  }

  if (log10column_H2 > log10column_H2_grid[n-1]){

    log10column_H2 = log10column_H2_grid[n-1];
  }



  /* Evaluate the spline function to get the bicubic interpolation (spline.cpp) */

  splin2( log10column_CO_grid, log10column_H2_grid, log10shield_CO_grid, d2log10shield, m, n,
          log10column_CO, log10column_H2, &log10shield );


  return pow(10.0, log10shield);

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

  tau_lambda = tau_visual * X_lambda(lambda);


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

  return dust_scatter;

}

/*-----------------------------------------------------------------------------------------------*/





/* X_lambda: Retuns ratio of optical depths at given lambda w.r.t. the visual wavelenght         */
/*-----------------------------------------------------------------------------------------------*/

double X_lambda(double lambda)
{

  /* Determine the ratio of the optical depth at a given wavelength to
     that at visual wavelength (λ=5500Å) using the extinction curve of
     Savage & Mathis (1979, ARA&A, 17, 73, Table 2) */

  long i;                                                                               /* index */

  const long n = 30;

  double lambda_grid[n] = {  910.0E0,   950.0E0,  1000.0E0,  1050.0E0, 1110.0E0, \
                            1180.0E0,  1250.0E0,  1390.0E0,  1490.0E0, 1600.0E0, \
                            1700.0E0,  1800.0E0,  1900.0E0,  2000.0E0, 2100.0E0, \
                            2190.0E0,  2300.0E0,  2400.0E0,  2500.0E0, 2740.0E0, \
                            3440.0E0,  4000.0E0,  4400.0E0,  5500.0E0, 7000.0E0, \
                            9000.0E0, 12500.0E0, 22000.0E0, 34000.0E0,    1.0E9   };

  double X_grid[n] = { 5.76E0, 5.18E0, 4.65E0, 4.16E0, 3.73E0, \
                       3.40E0, 3.11E0, 2.74E0, 2.63E0, 2.62E0, \
                       2.54E0, 2.50E0, 2.58E0, 2.78E0, 3.01E0, \
                       3.12E0, 2.86E0, 2.58E0, 2.35E0, 2.00E0, \
                       1.58E0, 1.42E0, 1.32E0, 1.00E0, 0.75E0, \
                       0.48E0, 0.28E0, 0.12E0, 0.05E0, 1.00E-50 };

  double yp0 = 1.0E30;                                               /* lower boundary condition */
  double ypn = 1.0E30;                                               /* upper boundary condition */

  double d2logX[n];                                   /* second order derivative of the function */

  double logX_result;                                      /* Resulting interpolated value for X */


  /* Scale the grids to get a better spline interpolation */
  /* The transformation was empirically determined */

  double loglambda = log(lambda);

  double loglambda_grid[n];
  double logX_grid[n];

  for (i=0; i<n; i++){

    loglambda_grid[i] = log(lambda_grid[i]);
    logX_grid[i]      = log(X_grid[i]);
  }


  /* Write the X_lambda values to a text file (for testing) */

  // FILE *xl = fopen("X_lambda.txt", "w");

  // if (xl == NULL){

  //     printf("Error opening file!\n");
  //     exit(1);
  //   }

  // for (int i=0; i<n; i++){

  //   fprintf(xl, "%lE\t%lE\n", lambda_grid[i], X_grid[i] );
  // }

  // fclose(xl);



  /* Calculate the cubic splines (spline.cpp) */

  spline(loglambda_grid, logX_grid, n, yp0, ypn, d2logX);


  /* Enforce the variables to be in the range of the interpolating function */

  if (loglambda < loglambda_grid[0]){

    loglambda = loglambda_grid[0];
  }

  if (loglambda > loglambda_grid[n-1]){

    loglambda = loglambda_grid[n-1];
  }


  /* Evaluate the spline function to get the interpolation (spline.cpp) */

  splint( loglambda_grid, logX_grid, d2logX, n, loglambda, &logX_result );


  return exp(logX_result);

}

/*-----------------------------------------------------------------------------------------------*/
