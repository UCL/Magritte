/* Frederik De Ceuster - University College London                                               */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/* heating: calculate the heating                                                                */
/*                                                                                               */
/* (based on read_species and read_rates in 3D-PDR)                                              */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/



#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include <string>
using namespace std;


/* heating: calculate the total heating                                                          */
/*-----------------------------------------------------------------------------------------------*/

double heating( GRIDPOINT *gridpoint, SPECIES *species, REACTIONS *reaction,
                double *temperature_gas, double *temperature_dust,
                double *UV_field, double v_turb )
{

  double heating_total                                                          /* total heating */

  double Habing_field = 1.68 * UV_field;                         /* UV radiation field in Habing */

  double electron_density = species[e_nr].abn[gridp] * gridpoint[gridp].density;    /* e density */



  /*   PHOTOELECTRIC DUST HEATING                                                                */
  /*_____________________________________________________________________________________________*/


  /*  Dust photoelectric heating using the treatment of Tielens & Hollenbach, 1985, ApJ, 291,
      722, which follows de Jong (1977,1980)
  
      The charge of a dust grain can be found by equating the rate of photo-ejection of
      electrons from the dust grain to the rate of recombination of electrons with the dust
      grain (Spitzer)
  
      The various parameter values are taken from Table 2 of the paper  */


  double heating_PhED;                                   /* resulting photoelectric dust heating */

  int iteration;                                /* iteration count for the Newton-Raphson solver */
  
  int max_iterations;                                            /* maximal number of iterations */

  const double precision = 1.0E-2;                     /* precision of the Newton-Raphson method */

  int e_nr = get_species_nr("e-");                      /* species nr corresponding to electrons */


  /* Parameters */

  double delta_d  = 1.0;
  double delta_UV = 1.8;
  
  double Y = 0.1;

  double hnu_d =  6.0;
  double hnu_H = 13.6;


  /* Derived parameters */

  double x_k = KB*temperature_gas/(hnu_H*EV) 
  double x_d = hnu_d/hnu_H;

  double gamma = 2.9E-4 * Y * sqrt(temperature_gas) * Habing_field / electron_density;

  double delta = x_k - x_d + gamma;


  /* Newton-Raphson iteration to find the zero of F(x) */

  F_x = 1.0;

  x = 0.5;

  iteration = 0;


  while( (iteration<max_iterations)  &&  (F_x > precision) ){

    x_0 = x - F(x,delta,gamma)/dF(x,delta);

    F_x = abs(x-x_0);

    x = x_0;

    iteration++;
  }


  heating_PhED = 2.7E-25 * delta_UV * delta_d * gridpoint[gridp].density * Y * Habing_field
                 * ( pow(1.0-x, 2)/x + x_k*(pow(X, 2) - 1.0)/pow(X, 2)  ) * metallicity;


  /*_____________________________________________________________________________________________*/





  /*   PHOTOELECTRIC PAH HEATING                                                                 */
  /*_____________________________________________________________________________________________*/


  /*  Grain + PAH photoelectric heating (MRN size distribution; r = 3-100 Å)

      Use the treatment of Bakes & Tielens (1994, ApJ, 427, 822) with the modifications suggested
      by Wolfire et al. (2003, ApJ, 587, 278) to account for the revised PAH abundance estimate
      from Spitzer data.

      See also:
      Wolfire et al. (1995, ApJ, 443, 152)
      Le Page, Snow & Bierbaum (2001, ApJS, 132, 233)  */


  double heating_PhEPAH;

  double phi_PAH = 1.0;

  double alpha = 0.944;
  
  double beta = 0.735 * pow(temperature_gas, 0.068);
  
  double delta = Habing_field * sqrt(temperature_gas) / (electron_density * phi_PAH);

  double epsilon = 4.87E-2/(1.0 + 4.0E-3*pow(delta, 0.73))
                   + 3.65E-2*pow(temperature_gas/1.0E4, 0.7)/(1.0 + 2.0E-4*delta);


  double PAH_heating = 1.3E-24 * epsilon * Habing_field * gridpoint[gridp].density;

  double PAH_cooling = 4.65E-30 * pow(temperature_gas, alpha) * pow(delta, beta) 
                       * electron_density * phi_PAH * gridpoint[gridp].density;


  /* Assume the PE heating rate scales linearly with metallicity */

  double heating_PhEPAH = (PAH_heating - PAH_cooling)*metallicity;


  /*_____________________________________________________________________________________________*/





  /*   WEINGARTNER HEATING                                                                       */
  /*_____________________________________________________________________________________________*/


  /*  Weingartner & Draine, 2001, ApJS, 134, 263

      Includes photoelectric heating due to PAHs, VSGs and larger grains, assumes a gas-to-dust
      mass ratio of 100:1  */


  double heating_Weingartner;

  double C0 = 5.72E+0;
  double C1 = 3.35E-2;
  double C2 = 7.08E-3;
  double C3 = 1.98E-2;
  double C4 = 4.95E-1;
  double C5 = 6.92E-1;
  double C6 = 5.20E-1;


  heating_Weingartner = 1.0E-26 * metallicity * (Habing_field * gridpoint[gridp].density) 
                        *(C0 + C1*pow(temperature_gas, C4))
                        /(1.0 + C2*pow(Habing_field*sqrt(temperature_gas)/electron_density,C5)
                        *(1.0 + C3*pow(Habing_field*sqrt(temperature_gas)/electron_density,C6) ));


  /*_____________________________________________________________________________________________*/





  /* Sum all contributions to the heating */

  heating_total = heating_PhED + heating_PhEPAH + heating_Weingartner;

  return heating_total; 

}

/*-----------------------------------------------------------------------------------------------*/





/* F: mathematical function used in photoelectric dust heating                                   */
/*-----------------------------------------------------------------------------------------------*/

double F(double x, double delta, double gamma)
{

  return pow(x,3) + delta*pow(x,2)-gamma;
}


/*-----------------------------------------------------------------------------------------------*/





/* dF: defivative w.r.t. x of the function F defined above                                       */
/*-----------------------------------------------------------------------------------------------*/

double dF(double x, double delta)
{

  return 3*pow(x,2) + 2*delta*x;
}

/*-----------------------------------------------------------------------------------------------*/
