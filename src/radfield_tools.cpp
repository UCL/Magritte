// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#include "declarations.hpp"
#include "radfield_tools.hpp"
#include "spline.hpp"



#define IND(r,c) ((c)+(r)*n)



// self_shielding_H2: Returns H2 self-shielding function
// -----------------------------------------------------

double self_shielding_H2 (double column_H2, double doppler_width, double radiation_width)
{

  // Following Federman, Glassgold & Kwan (1979, ApJ, 227, 466)

  double J_D;              // Doppler contribution to self-shielding
  double J_R;              // Radiative contribution to self-shielding
  double self_shielding;   // total self-shielding function


  // Calculate optical depth at line centre
  // = N(H2)*f_para*(πe^2/mc)*f/(√πß) ≈ N(H2)*f_para*(1.5E-2)*f/ß

  double frac_H2_para = 0.5;              // (assume H2_ortho / H2_para ratio = 1)
  double f_osc        = 1.0E-2;           // Oscillator strength of a typical transition
  double PIe2_mc      = 1.497358985E-2;   // PI e^2 / mc, with electron charge (e) and mass (m)


 // Optical depth at line centre
 // (parameter tau_D (eq. A7) in Federman's paper)

  double tau_D = column_H2 * frac_H2_para * PIe2_mc * f_osc / doppler_width;


  // Calculate Doppler core contribution to self-shielding (JD)
  // (parameter JD (eq. A8) in Federman's paper)

  if      (tau_D == 0.0)
  {
    J_D = 1.0;
  }
  else if (tau_D < 2.0)
  {
    J_D = exp(-0.666666667*tau_D);
  }
  else if (tau_D < 10.0)
  {
    J_D = 0.638 * pow(tau_D, -1.25);
  }
  else if (tau_D < 100.0)
  {
    J_D = 0.505 * pow(tau_D, -1.15);
  }
  else
  {
    J_D = 0.344 * pow(tau_D, -1.0667);
  }


  // Calculate radiative wing contribution to self-shielding (JR)
  // (parameter JR (eq. A9) in Federman's paper)

  if (radiation_width == 0.0)
  {
    J_R = 0.0;
  }
  else
  {
    double sqrt_PI = 1.772453851;                           // square root of PI
    double r  = radiation_width / (sqrt_PI*doppler_width);  // (equation A2 in Federman's paper)
    double t1 = 3.02 * pow(r*1.0E3,-0.064);                 // (equation A6 in Federman's paper)
    double u1 = sqrt(tau_D*r) / t1;                         // (equation A6 in Federman's paper)

    J_R = r / ( t1 * sqrt(PI/4.0 + u1*u1) );
  }


  // Calculate total self-shielding function

  return self_shielding = J_D + J_R;

}




/* self_shielding_CO: Returns CO self-shielding function                                         */
/*-----------------------------------------------------------------------------------------------*/

double self_shielding_CO( double column_CO, double column_H2 )
{

  /*  12CO line shielding, using the computed values listed in
      van Dishoeck & Black (1988, ApJ, 334, 771, Table 5)

      Appropriate shielding factors are determined by performing a 2-dimensional spline
      interpolation over the values listed in Table 5 of van Dishoeck & Black, which include
      contributions from self-shielding and H2 screening  */


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

  double d2log10shield[m*n];


  /* Write the shield_CO values to a text file (for testing) */

  // FILE *sCO = fopen("output/self_shielding_CO_table.txt", "w");
  //
  // if (sCO == NULL){
  //
  //     printf("Error opening file!\n");
  //     exit(1);
  //   }
  //
  // for (long i=0; i<m; i++){
  //
  //   for (long j=0; j<n; j++){
  //
  //     fprintf(sCO, "%lE\t%lE\t%lE\n", log10column_CO_grid[i],
  //                                     log10column_H2_grid[j],
  //                                     log10shield_CO_grid[IND(i,j)] );
  //   }
  // }
  //
  // fclose(sCO);



  /* Calculate the splines for the rows (spline.cpp) */

  splie2( log10column_CO_grid, log10column_H2_grid, log10shield_CO_grid, m, n, d2log10shield );



  /* Scale the variables to get a better spline interpolation */

  double log10column_CO = log10(column_CO + 1.0);
  double log10column_H2 = log10(column_H2 + 1.0);



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


  /* Coefficients in equation (1) in Wagenblast & Hartquist 1989                                 */
  /*     A(0)    = a(0)*exp(-k(0)*tau)                                                           */
  /*             = relative intensity decrease for 0 < tau < 1                                   */
  /*     A(I)    = ∑ a(i)*exp(-k(i)*tau) for i=1,5                                               */
  /*               relative intensity decrease for tau ≥ 1                                       */

  double A[6] = { 1.000,  2.006, -1.438, 0.7364, -0.5076, -0.0592 };


  /*     K(0)    = see A0                                                                        */
  /*     K(I)    = see A(I)                                                                      */

  double k[6] = { 0.7514, 0.8490, 1.013, 1.282,   2.005,   5.832 };


  double dust_scatter = 0.0;                            /* attenuation due to scattering by dust */


  /* Calculate the optical depth at visual wavelength */

  double tau_visual = AV_ray / 1.086;


  /* Convert the optical depth to that at the desired wavelength */

  double tau_lambda = tau_visual * X_lambda(lambda);


  /* Calculate the attenuation due to scattering by dust */
  /* equation (1) in Wagenblast & Hartquist 1989 */

  if ( tau_lambda < 1.0 ){

    double exponent = tau_lambda * k[0];

    if ( exponent < 100.0 ){

      dust_scatter = A[0] * exp(-exponent);
    }
  }

  else {

    for (int i=1; i<6; i++){

      double exponent = tau_lambda * k[i];

      if ( exponent < 100.0 ){

        dust_scatter = dust_scatter + A[i]*exp(-exponent);
      }
    }
  }

  return dust_scatter;

}

/*-----------------------------------------------------------------------------------------------*/





// X_lambda: Retuns ratio of optical depths at given lambda w.r.t. visual wavelenght
// ---------------------------------------------------------------------------------

double X_lambda (double lambda)
{

  /* Determine the ratio of the optical depth at a given wavelength to
     that at visual wavelength (λ=5500Å) using the extinction curve of
     Savage & Mathis (1979, ARA&A, 17, 73, Table 2) */


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
                       0.48E0, 0.28E0, 0.12E0, 0.05E0, 1.0E-99 };

  double yp0 = 1.0E31;   // lower boundary condition
  double ypn = 1.0E31;   // upper boundary condition

  double d2logX[n];      // second order derivative of the function

  double logX_result;    // Resulting interpolated value for X


  // Scale grids to get better spline interpolation

  double loglambda = log(lambda);

  double loglambda_grid[n];
  double logX_grid[n];

  for (int i = 0; i < n; i++)
  {
    loglambda_grid[i] = log(lambda_grid[i]);
    logX_grid[i]      = log(X_grid[i]);
  }


  // Calculate cubic splines (spline.cpp)

  spline (loglambda_grid, logX_grid, n, yp0, ypn, d2logX);


  // Enforce variables to be in range of interpolating function

  if (loglambda < loglambda_grid[0])
  {
    loglambda = loglambda_grid[0];
  }

  if (loglambda > loglambda_grid[n-1])
  {
    loglambda = loglambda_grid[n-1];
  }


  // Evaluate spline function to get interpolation (spline.cpp)

  splint (loglambda_grid, logX_grid, d2logX, n, loglambda, &logX_result);


  return exp(logX_result);

}
