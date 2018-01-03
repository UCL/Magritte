// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <iostream>

#include "../parameters.hpp"
#include "Magritte_config.hpp"
#include "declarations.hpp"

#include "heating.hpp"


// heating: calculate total heating
// --------------------------------

double heating (long ncells, CELL *cell, long gridp, double *UV_field, double* heating_components)
{

  double Habing_field = 1.68 * UV_field[gridp];            // UV radiation field in Habing

  double electron_density = species[e_nr].abn[gridp] * cell[gridp].density;   // e density




  // PHOTOELECTRIC DUST HEATING
  // __________________________


  /*  Dust photoelectric heating using the treatment of Tielens & Hollenbach, 1985, ApJ, 291,
      722, which follows de Jong (1977,1980)

      The charge of a dust grain can be found by equating the rate of photo-ejection of
      electrons from the dust grain to the rate of recombination of electrons with the dust
      grain (Spitzer)

      The various parameter values are taken from Table 2 of the paper  */

  const double precision = 1.0E-2;   // precision of Newton-Raphson method


  // Parameters

  double Delta_d  = 1.0;
  double Delta_UV = 1.8;

  double Y = 0.1;

  double hnu_d =  6.0;
  double hnu_H = 13.6;


  // Derived parameters

  double x_k = KB*cell[gridp].temperature.gas/(hnu_H*EV);
  double x_d = hnu_d/hnu_H;

  double gamma = 2.9E-4 * Y * sqrt(cell[gridp].temperature.gas) * Habing_field / electron_density;

  double Delta = x_k - x_d + gamma;


  // Newton-Raphson iteration to find root of F(x)

  double F_x = 1.0;

  double x = 0.5;

  int iteration = 0;   // iteration count for Newton-Raphson solver



  while( (iteration < 100) && (F_x > precision) )
  {
    double x_0 = x - F(x,Delta,gamma)/dF(x,Delta);

    F_x = fabs(x-x_0);

    x = x_0;

    iteration++;
  }


  double heating_dust = 2.7E-25 * Delta_UV * Delta_d * cell[gridp].density * Y * Habing_field
                        * ( pow(1.0-x, 2)/x + x_k*(pow(x, 2) - 1.0)/pow(x, 2)  ) * METALLICITY;

  heating_components[0] = heating_dust;




  // PHOTOELECTRIC PAH HEATING
  // _________________________


  /*  Grain + PAH photoelectric heating (MRN size distribution; r = 3-100 Å)

      Use the treatment of Bakes & Tielens (1994, ApJ, 427, 822) with the modifications suggested
      by Wolfire et al. (2003, ApJ, 587, 278) to account for the revised PAH abundance estimate
      from Spitzer data.

      See also:
      Wolfire et al. (1995, ApJ, 443, 152)
      Le Page, Snow & Bierbaum (2001, ApJS, 132, 233)  */


  double phi_PAH = 1.0;

  double alpha = 0.944;

  double beta = 0.735 / pow(cell[gridp].temperature.gas, 0.068);

  double delta = Habing_field * sqrt(cell[gridp].temperature.gas) / (electron_density * phi_PAH);

  double epsilon = 4.87E-2/(1.0 + 4.0E-3*pow(delta, 0.73))
                   + 3.65E-2*pow(cell[gridp].temperature.gas/1.0E4, 0.7)/(1.0 + 2.0E-4*delta);


  double PAH_heating = 1.3E-24 * epsilon * Habing_field * cell[gridp].density;

  double PAH_cooling = 4.65E-30 * pow(cell[gridp].temperature.gas, alpha) * pow(delta, beta)
                       * electron_density * phi_PAH * cell[gridp].density;


  // Assume PE heating rate scales linearly with METALLICITY

  double heating_PAH = (PAH_heating - PAH_cooling)*METALLICITY;

  heating_components[1] = heating_PAH;




  // WEINGARTNER HEATING
  // ___________________


  /*  Weingartner & Draine, 2001, ApJS, 134, 263

      Includes photoelectric heating due to PAHs, VSGs and larger grains, assumes a gas-to-dust
      mass ratio of 100:1  */


  double C0 = 5.72E+0;
  double C1 = 3.45E-2;
  double C2 = 7.08E-3;
  double C3 = 1.98E-2;
  double C4 = 4.95E-1;
  double C5 = 6.92E-1;
  double C6 = 5.20E-1;


  double heating_Weingartner
        = 1.0E-26 * METALLICITY * (Habing_field * cell[gridp].density)
          * ( C0 + C1*pow(cell[gridp].temperature.gas, C4) )
          / ( 1.0 + C2*pow(Habing_field * sqrt(cell[gridp].temperature.gas) / electron_density, C5)
          * ( 1.0 + C3*pow(Habing_field * sqrt(cell[gridp].temperature.gas) / electron_density, C6) ) );

  heating_components[2] = heating_Weingartner;




  // CARBON PHOTOIONIZATION HEATING
  // ______________________________


  /*  Contributes 1 eV on average per carbon ionization
      Use the C photoionization rate determined in calc_reac_rates_rad.cpp [units: s^-1]  */


  double heating_C_ionization = (1.0*EV) * reaction[C_ionization_nr].k[gridp]
                                * species[C_nr].abn[gridp] * cell[gridp].density;

  heating_components[3] = heating_C_ionization;




  // H2 FORMATION HEATING
  // ____________________


  /*  Hollenbach & Tielens, Review of Modern Physics, 1999, 71, 173

      Assume 1.5 eV liberated as heat during H2 formation
      Use the H2 formation rate determined in calc_reac_rates.cpp [units: cm^3.s^-1]  */


  double heating_H2_formation = (1.5*EV) * reaction[H2_formation_nr].k[gridp]
                                * cell[gridp].density * species[H_nr].abn[gridp]
                                * cell[gridp].density;

  heating_components[4] = heating_H2_formation;




  // H2 PHOTODISSOCIATION HEATING
  // ____________________________


  /*  Contributes 0.4 eV on average per photodissociated molecule
      Use H2 photodissociation rate determined in calc_reac_rates_rad.cpp [units: s^-1]  */


  double heating_H2_photodissociation = (0.4*EV) * reaction[H2_photodissociation_nr].k[gridp]
                                        * species[H2_nr].abn[gridp] * cell[gridp].density;

  heating_components[5] = heating_H2_photodissociation;




  // H2 FUV PUMPING HEATING
  // ______________________


  /*  Hollenbach & McKee (1979)

      Contributes 2.2 eV on average per vibrationally excited H2* molecule
      Use H2 photodissociation rate determined in calc_reac_rates_rad.cpp [units: s^-1]
      Use H2 critical density calculation from Hollenbach & McKee (1979)  */


  double critical_density
          = 1.0E6 / sqrt(cell[gridp].temperature.gas)
            /( 1.6 * species[H_nr].abn[gridp] * exp(-pow(400.0/cell[gridp].temperature.gas, 2))
               + 1.4 * species[H2_nr].abn[gridp] * exp(-18100.0/(1200.0+cell[gridp].temperature.gas)) );

  double heating_H2_FUV_pumping = (2.2*EV) * 9.0 * reaction[H2_photodissociation_nr].k[gridp]
                                  * species[H2_nr].abn[gridp] * cell[gridp].density
                                  / (1.0 + critical_density/cell[gridp].density);

  heating_components[6] = heating_H2_FUV_pumping;




  // COSMIC-RAY IONIZATION HEATING
  // _____________________________


  /*  Tielens & Hollenbach, 1985, ApJ, 291, 772

      Contributes 8.0 eV deposited per primary ionization (plus some from He ionization)

      See also:
      Shull & Van Steenberg, 1985, ApJ, 298, 268
      Clavel et al. (1978), Kamp & van Zadelhoff (2001)  */


  double heating_cosmic_rays = (9.4*EV) * (1.3E-17*ZETA)
                               * species[H2_nr].abn[gridp] * cell[gridp].density;

  heating_components[7] = heating_cosmic_rays;




  // SUPERSONIC TURBULENT DECAY HEATING
  // __________________________________


  /*  Most relevant for the inner parsecs of galaxies Black, Interstellar Processes, 1987, p731

      See also:
      Rodriguez-Fernandez et al., 2001, A&A, 365, 174  */


  double l_turb = 5.0;   // turbulent length scale (typical value) in parsec


  double heating_turbulent = 3.5E-28*pow(V_TURB/1.0E5, 3)*(1.0/l_turb)*cell[gridp].density;

  heating_components[8] = heating_turbulent;




  // EXOTHERMAL CHEMICAL REACTION HEATING
  // ____________________________________


  /*  Clavel et al., 1978, A&A, 65, 435

      Recombination reactions: HCO+ (7.51 eV); H3+ (4.76+9.23 eV); H3O+ (1.16+5.63+6.27 eV)

      Ion-neutral reactions  : He+ + H2 (6.51 eV); He+ + CO (2.22 eV)

      For each reaction, the heating rate should be: n(1) * n(2) * K * E with n(1) and n(2)
      the densities, K the rate coefficient [cm^3.s^-1], and E the energy [erg]  */


  // For so-called REDUCED NETWORK of 3D-PDR

  double heating_chemical
                   = species[H2x_nr].abn[gridp] * cell[gridp].density      // H2+  +  e-
                     * electron_density
                     * reaction[215].k[gridp] * (10.9*EV)

                     + species[H2x_nr].abn[gridp] * cell[gridp].density    // H2+  +  H
                       * species[H_nr].abn[gridp] * cell[gridp].density
                       * reaction[154].k[gridp] * (0.94*EV)

                     + species[HCOx_nr].abn[gridp] * cell[gridp].density   // HCO+  +  e-
                       * electron_density
                       * reaction[239].k[gridp] * (7.51*EV)

                     + species[H3x_nr].abn[gridp] * cell[gridp].density    // H3+  +  e-
                       * electron_density
                       * ( reaction[216].k[gridp] * (4.76*EV) + reaction[217].k[gridp] * (9.23*EV) )

                     + species[H3Ox_nr].abn[gridp]*cell[gridp].density     // H3O+  + e-
                       * electron_density
                       * ( reaction[235].k[gridp] * (1.16*EV) + reaction[236].k[gridp] * (5.63*EV)
                           + reaction[237].k[gridp] * (6.27*EV) )

                     + species[Hex_nr].abn[gridp] * cell[gridp].density    // He+  + H2
                       * species[H2_nr].abn[gridp] * cell[gridp].density
                       * ( reaction[49].k[gridp] * (6.51*EV) + reaction[169].k[gridp] * (6.51*EV) )

                     + species[Hex_nr].abn[gridp] * cell[gridp].density    // He+  + CO
                       * species[CO_nr].abn[gridp] *cell[gridp].density
                       * ( reaction[88].k[gridp] * (2.22*EV) + reaction[89].k[gridp] * (2.22*EV)
                           + reaction[90].k[gridp] * (2.22*EV) );

  heating_components[9] = heating_chemical;




  // GAS-GRAIN COLLISIONAL HEATING
  // _____________________________


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


  double radius_grain = 1.0E-7;                                     /* radius of the dust grains */
  // double radius_grain = 1.0E-5;                                     /* radius of the dust grains */


  double accommodation = 0.1
                         + 0.35*exp(-sqrt((cell[gridp].temperature.gas+cell[gridp].temperature.dust)/5.0E2));

  double density_grain = 1.998E-12 * cell[gridp].density * METALLICITY * 100.0 / GAS_TO_DUST;

  double cross_section_grain = PI * pow(radius_grain, 2);

  double heating_gas_grain = 4.003E-12 * cell[gridp].density * density_grain
                             * cross_section_grain * accommodation * sqrt(cell[gridp].temperature.gas)
                             * (cell[gridp].temperature.dust - cell[gridp].temperature.gas);

  heating_components[10] = heating_gas_grain;




  // Sum all contributions to the heating

  double heating_total = /*heating_dust*/
                         + heating_PAH
                         /*+ heating_Weingartner*/
                         + heating_C_ionization
                         + heating_H2_formation
                         + heating_H2_photodissociation
                         + heating_H2_FUV_pumping
                         + heating_cosmic_rays
                         + heating_turbulent
                         + heating_chemical
                         + heating_gas_grain;

  heating_components[11] = heating_total;

  // cout << "dust        " << heating_dust << "\n";
  // cout << "PAH         " << heating_PAH << "\n";
  // cout << "Weingartner " << heating_Weingartner << "\n";
  // cout << "C_ion       " << heating_C_ionization << "\n";
  // cout << "H2_phot     " << heating_H2_photodissociation << "\n";
  // cout << "H2_FUV      " << heating_H2_FUV_pumping << "\n";
  // cout << "CR          " << heating_cosmic_rays << "\n";
  // cout << "tur         " << heating_turbulent << "\n";
  // cout << "chem        " << heating_chemical << "\n";
  // cout << "gas-grain   " << heating_gas_grain << "\n";

  return heating_total;

}




// F: mathematical function used in photoelectric dust heating
// -----------------------------------------------------------

double F (double x, double delta, double gamma)
{

  return pow(x,3) + delta*pow(x,2) - gamma;
}




// dF: defivative w.r.t. x of function F defined above
// ---------------------------------------------------

double dF (double x, double delta)
{

  return 3.0*pow(x,2) + 2.0*delta*x;
}
