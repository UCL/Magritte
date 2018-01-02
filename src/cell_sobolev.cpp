// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "../parameters.hpp"
#include "Magritte_config.hpp"
#include "declarations.hpp"

#if (CELL_BASED)

#include "cell_sobolev.hpp"
#include "ray_tracing.hpp"


// sobolev: calculate mean intensity using LVG approximation and escape probabilities
// ----------------------------------------------------------------------------------

int cell_sobolev( CELL *cell, double *mean_intensity, double *Lambda_diagonal,
                  double *mean_intensity_eff, double *source, double *opacity, double *frequency,
                  double *temperature_gas, double *temperature_dust, int *irad, int*jrad,
                  long origin, int lspec, int kr )
{

  long m_ij = LSPECGRIDRAD(lspec,origin,kr);   // mean_intensity, S and opacity index

  int i = irad[LSPECRAD(lspec,kr)];            // i level index corresponding to transition kr
  int j = jrad[LSPECRAD(lspec,kr)];            // j level index corresponding to transition kr

  long b_ij = LSPECLEVLEV(lspec,i,j);          // frequency index


  double speed_width = sqrt(8.0*KB*temperature_gas[origin]/PI/MP + pow(V_TURB, 2));

  double escape_probability = 0.0;             // escape probability from Sobolev approximation


  // DO THE RADIATIVE TRANSFER
  // _ _ _ _ _ _ _ _ _ _ _ _ _


  // For half of the rays (only half is needed since we also consider the antipodals)

  for (long r = 0; r < NRAYS/2; r++)
  {

    // Get the antipodal ray for r

    long ar = antipod[r];


    // Fill the source function and the optical depth increment along ray r

    double tau_r  = 0.0;
    double tau_ar = 0.0;


    // Walk along antipodal ray (ar) of r

    {
      double Z    = 0.0;
      double dZ   = 0.0;
      double dtau = 0.0;

      long current = origin;
      long next    = next_cell (NCELLS, cell, origin, ar, Z, current, &dZ);


      while ( (next != NCELLS) && (tau_ar < TAU_MAX) )
      {
        long s_n = LSPECGRIDRAD(lspec,next,kr);
        long s_c = LSPECGRIDRAD(lspec,current,kr);

        dtau = dZ * PC * (opacity[s_n] + opacity[s_c]) / 2.0;

        tau_ar = tau_ar + dtau;
        Z      = Z + dZ;

        current = next;
        next    = next_cell (NCELLS, cell, origin, ar, Z, current, &dZ);
      }
    }


    // Calculate ar's contribution to escape probability

    tau_ar = CC / frequency[b_ij] / speed_width * tau_ar;


    if (tau_ar < -5.0)
    {
      escape_probability = escape_probability + (1 - exp(5.0)) / (-5.0);
    }

    else if (fabs(tau_ar) < 1.0E-8)
    {
      escape_probability = escape_probability + 1.0;
    }

    else
    {
      escape_probability = escape_probability + (1.0 - exp(-tau_ar)) / tau_ar;
    }


    // Walk along ray r itself

    {
      double Z    = 0.0;
      double dZ   = 0.0;
      double dtau = 0.0;

      long current = origin;
      long next    = next_cell (NCELLS, cell, origin, r, Z, current, &dZ);


      while ( (next != NCELLS) && (tau_r < TAU_MAX) )
      {
        long s_n = LSPECGRIDRAD(lspec,next,kr);
        long s_c = LSPECGRIDRAD(lspec,current,kr);

        dtau = dZ * PC * (opacity[s_n] + opacity[s_c]) / 2.0;

        tau_r = tau_r + dtau;
        Z     = Z + dZ;

        current = next;
        next    = next_cell (NCELLS, cell, origin, r, Z, current, &dZ);
      }
    }


    // Calculate r's contribution to escape probability

    tau_r = CC / frequency[b_ij] / speed_width * tau_r;


    if (tau_r < -5.0)
    {
      escape_probability = escape_probability + (1 - exp(5.0)) / (-5.0);
    }

    else if (fabs(tau_r) < 1.0E-8)
    {
      escape_probability = escape_probability + 1.0;
    }

    else
    {
      escape_probability = escape_probability + (1.0 - exp(-tau_r)) / tau_r;
    }

  } // end of r loop over half of the rays


  escape_probability = escape_probability / NRAYS;

  printf("esc prob %lE\n", escape_probability);


  // ADD CONTINUUM RADIATION (due to dust and CMB)
  // _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _


  // Continuum radiation is assumed to be local

  double factor          = 2.0*HH*pow(frequency[b_ij],3)/pow(CC,2);

  double rho_grain       = 2.0;

  double ngrain          = 2.0E-12*cell[origin].density*METALLICITY*100.0/GAS_TO_DUST;

  double emissivity_dust = rho_grain*ngrain*0.01*1.3*frequency[b_ij]/3.0E11;

  double Planck_dust     = 1.0 / (exp(HH*frequency[b_ij]/KB/temperature_dust[origin]) - 1.0);

  double Planck_CMB      = 1.0 / (exp(HH*frequency[b_ij]/KB/T_CMB) - 1.0);


  double continuum_mean_intensity = factor * (Planck_CMB + emissivity_dust*Planck_dust);


  mean_intensity[m_ij] = (1.0 - escape_probability) * source[m_ij]
                         + escape_probability * continuum_mean_intensity;


  if (ACCELERATION_APPROX_LAMBDA)
  {
    Lambda_diagonal[m_ij]    = (1.0 - escape_probability);

    mean_intensity_eff[m_ij] = escape_probability * continuum_mean_intensity;
  }

  else
  {
    Lambda_diagonal[m_ij]    = 0.0;

    mean_intensity_eff[m_ij] = mean_intensity[m_ij];
  }


  return(0);

}


#endif // if CELL_BASED
