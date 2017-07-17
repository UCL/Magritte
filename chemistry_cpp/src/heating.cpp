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

  int e_nr = get_species_nr("e-");                      /* species nr corresponding to electrons */

  int H2_nr = get_species_nr("H2");                            /* species nr corresponding to H2 */

  int C_nr = get_species_nr("C");                               /* species nr corresponding to C */

  int H_nr = get_species_nr("H");                               /* species nr corresponding to H */

  int H2x_nr = get_species_nr("H2+");                         /* species nr corresponding to H2+ */
 
  int HCOx_nr = get_species_nr("HCO+");                      /* species nr corresponding to HCO+ */

  int H3x_nr = get_species_nr("H3+");                         /* species nr corresponding to H3+ */

  int H3Ox_nr = get_species_nr("H3O+");                      /* species nr corresponding to H3O+ */

  int Hex_nr = get_species_nr("He+");                         /* species nr corresponding to He+ */

  int CO_nr = get_species_nr("CO");                            /* species nr corresponding to CO */


  double electron_density = species[e_nr].abn[gridp] * gridpoint[gridp].density;    /* e density */



  /*   PHOTOELECTRIC DUST HEATING                                                                */
  /*_____________________________________________________________________________________________*/


  /*  Dust photoelectric heating using the treatment of Tielens & Hollenbach, 1985, ApJ, 291,
      722, which follows de Jong (1977,1980)
  
      The charge of a dust grain can be found by equating the rate of photo-ejection of
      electrons from the dust grain to the rate of recombination of electrons with the dust
      grain (Spitzer)
  
      The various parameter values are taken from Table 2 of the paper  */


  double heating_dust;                                   /* resulting photoelectric dust heating */

  int iteration;                                /* iteration count for the Newton-Raphson solver */
  
  int max_iterations;                                            /* maximal number of iterations */

  const double precision = 1.0E-2;                     /* precision of the Newton-Raphson method */



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

  double F(double x, double delta, double gamma);

  double dF(double x, double delta);


  while( (iteration<max_iterations)  &&  (F_x > precision) ){

    x_0 = x - F(x,delta,gamma)/dF(x,delta);

    F_x = abs(x-x_0);

    x = x_0;

    iteration++;
  }


  heating_dust = 2.7E-25 * delta_UV * delta_d * gridpoint[gridp].density * Y * Habing_field
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


  double heating_PAH;

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

  double heating_PAH = (PAH_heating - PAH_cooling)*metallicity;


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





  /*   CARBON PHOTOIONIZATION HEATING                                                            */
  /*_____________________________________________________________________________________________*/


  /*  Contributes 1 eV on average per carbon ionization
      Use the C photoionization rate determined in rate_calculations_radfield.cpp [units: s^-1]  */


  double heating_C_ionization;


  heating_C_ionization = (1.0*EV) * reaction[C_ionization_nr].k
                         * species[C_nr].abn[gridp] * gridpoint[gridp].density;


  /*_____________________________________________________________________________________________*/





  /*   H2 FORMATION HEATING                                                                      */
  /*_____________________________________________________________________________________________*/


  /*  Hollenbach & Tielens, Review of Modern Physics, 1999, 71, 173

      Assume 1.5 eV liberated as heat during H2 formation
      Use the H2 formation rate determined in rate_calculations.cpp [units: cm^3.s^-1]  */


  double heating_H2_formation;


  heating_H2_formation = (1.5*EV) * reaction[H2_formation] * gridpoint[gridp].density
                         * species[H2_nr].abn[gridp] * gridpoint[gridp].density;


  /*_____________________________________________________________________________________________*/





  /*   H2 PHOTODISSOCIATION HEATING                                                              */
  /*_____________________________________________________________________________________________*/


  /*  Contributes 0.4 eV on average per photodissociated molecule
      Use H2 photodissociation rate determined in rate_calculations_radfield.cpp [units: s^-1]  */


  double heating_H2_photodissociation;


  heating_H2_photodissociation = (0.4*EV) * reaction[H2_photodissociation_nr].k
                                 * species[H2_nr].abn[gridp] * gridpoint[gridp].density;


  /*_____________________________________________________________________________________________*/





  /*   H2 FUV PUMPING HEATING                                                                    */
  /*_____________________________________________________________________________________________*/


  /*  Hollenbach & McKee (1979)

      Contributes 2.2 eV on average per vibrationally excited H2* molecule
      Use H2 photodissociation rate determined in rate_calculations_radfield.cpp [units: s^-1]
      Use H2 critical density calculation from Hollenbach & McKee (1979)  */


  double heating_H2_FUV_pumping;

  double critical_density = 1.0E6 / sqrt(temperature_gas)
                      / ( 1.6*species[H_nr].abn[gridp]*exp(-pow(400.0/temperature_gas, 2))
                          + 1.4*species[H2_nr].abn[gridp]*exp(-18100.0/(1200.0+temperature_gas)) )

  heating_H2_FUV_pumping = (2.2*EV) * 9.0 * reaction[H2_photodissociation_nr].k
                           * species[H2_nr].abn[gridp] * gridpoint[gridp].density
                           / (1.0 + critical_density/gridpoint[gridp].density);


  /*_____________________________________________________________________________________________*/





  /*   COSMIC-RAY IONIZATION HEATING                                                             */
  /*_____________________________________________________________________________________________*/


  /*  Tielens & Hollenbach, 1985, ApJ, 291, 772

      Contributes 8.0 eV deposited per primary ionization (plus some from He ionization)

      See also:
      Shull & Van Steenberg, 1985, ApJ, 298, 268
      Clavel et al. (1978), Kamp & van Zadelhoff (2001)  */


  double heating_cosmic_rays;

  double zeta = 1.0;                                                      /* cosmic ray variable */


  heating_cosmic_rays = (9.4*EV) * (1.3E-17*zeta) * species[H2_nr].abn * gridpoint[gridp].density;


  /*_____________________________________________________________________________________________*/





  /*   SUPERSONIC TURBULENT DECAY HEATING                                                        */
  /*_____________________________________________________________________________________________*/


  /*  Most relevant for the inner parsecs of galaxies Black, Interstellar Processes, 1987, p731

      See also:
      Rodriguez-Fernandez et al., 2001, A&A, 365, 174  */


  double heating_turbulent;

  double l_turb = 5.0                        /* turbulent length scale (typical value) in parsec */


  heating_turbulent = 3.5E-28 * pow(v_turb/1.0E5, 3) * (1.0/l_turb) * gridpoint[gridp].density;


  /*_____________________________________________________________________________________________*/





  /*   EXOTHERMAL CHEMICAL REACTION HEATING                                                      */
  /*_____________________________________________________________________________________________*/


  /*  Clavel et al., 1978, A&A, 65, 435

      Recombination reactions: HCO+ (7.51 eV); H3+ (4.76+9.23 eV); H3O+ (1.16+5.63+6.27 eV)
      
      Ion-neutral reactions  : He+ + H2 (6.51 eV); He+ + CO (2.22 eV)
      
      For each reaction, the heating rate should be: n(1) * n(2) * K * E with n(1) and n(2)
      the densities, K the rate coefficient [cm^3.s^-1], and E the energy [erg]  */

  double heating_chemical;


  /* For the so-called REDUCED NETWORK of 3D-PDR */

  heating_chemical = species[H2x_nr].abn[gridp] * gridpoint[gridp].density         /* H2+  +  e- */
                     * electron_density
                     * reaction[216].k * (10.9*EV)

                     + species[H2x_nr].abn[gridp] * gridpoint[gridp].density        /* H2+  +  H */
                       * species[H_nr].abn[gridp] * gridpoint[gridp].density
                       * reaction[155].k * (0.94*EV)

                     + species[HCOx_nr].abn[gridp] * gridpoint[gridp].density     /* HCO+  +  e- */
                       * electron_density
                       * reaction[240].k * (7.51*EV) 

                     + species[H3x_nr].abn[gridp] * gridpoint[gridp].density       /* H3+  +  e- */
                       * electron_density 
                       * ( reaction[217].k * (4.76*EV) + reaction[218].k * (9.23*EV) )

                     + species[H3Ox_nr].abn[gridp]*gridpoint[gridp].density       /* H3O+  +  e- */
                       * electron_density
                       * ( reaction[236].k * (1.16*EV) + reaction[237].k * (5.63*EV)
                           + reaction[238].k * (6.27*EV) )

                     + species[Hex_nr].abn[gridp] * gridpoint[gridp].density        /* He+  + H2 */
                       * species[H2_nr].abn[gridp] * gridpoint[gridp]
                       * ( reaction[50].k * (6.51*EV) + reaction[170] * (6.51*EV) )                 
                    
                     + species[Hex_nr].abn[gridp] * gridpoint[gridp].density       /* He+  +  CO */
                       * species[CO_nr].abn[gridp] *gridpoint[gridp].density
                       * ( reaction[89].k * (2.22*EV) + reaction[90].k * (2.22*EV)
                           + reaction[91] * (2.22*EV) );


  /*_____________________________________________________________________________________________*/





  /*   GAS-GRAIN COLLISIONAL HEATING                                                             */
  /*_____________________________________________________________________________________________*/


  /*  Use the treatment of Burke & Hollenbach, 1983, ApJ, 265, 223, and
      accommodation fitting formula of Groenewegen, 1994, A&A, 290, 531
 
      Other relevant references:
 
      Hollenbach & McKee, 1979, ApJS, 41,555
      Tielens & Hollenbach, 1985, ApJ, 291,722
      Goldsmith, 2001, ApJ, 557, 736
 
      This process is insignificant for the energy balance of the dust but can influence the gas
      temperature. If the dust temperature is lower than the gas temperature, this becomes a
      cooling mechanism
 
      In Burke & Hollenbach (1983) the factor:
 
      (8*kb/(pi*mass_proton))**0.5*2*kb = 4.003D-12

      This value has been used in the expression below  */


  double heating_gas_grain;

  double accommodation = 0.35 * exp(-sqrt((temperature_gas+temperature_dust)/5.0E2)) + 0.1;

  double density_grain = 1.998E-12 * gridpoint[gridp].density * metallicity * 100.0 / gas_to_dust;

  double cross_section_grain = PI * pow(radius_grain, 2);

  heating_gas_grain = 4.003E-12 * gridpoint[gridp].density * density_grain * cross_section_grain
                      * accommodation * sqrt(temperature_gas)
                      * (temperature_gas - temperature_dust);


  /*_____________________________________________________________________________________________*/





  /* Sum all contributions to the heating */

  heating_total = heating_dust + heating_PAH + heating_Weingartner + heating_C_ionization
                  + heating_H2_photodissociation + heating_H2_FUV_pumping + heating_cosmic_rays;

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
