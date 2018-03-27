// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <iostream>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "declarations.hpp"
#include "sobolev.hpp"
#include "ray_tracing.hpp"


// sobolev: calculate mean intensity using LVG approximation and escape probabilities
// ----------------------------------------------------------------------------------

int sobolev (long ncells, CELLS *cells, RAYS rays, LINES lines,
             double *Lambda_diagonal, double *mean_intensity_eff,
             double *source, double *opacity, long origin, int ls, int kr)
{

  long m_ij  = LSPECGRIDRAD(ls,origin,kr);   // mean_intensity, S and opacity index
  long mm_ij = LSPECRAD(ls, kr);             // mean_intensity, S and opacity index


  int i = lines.irad[LSPECRAD(ls,kr)];   // i level index corresponding to transition kr
  int j = lines.jrad[LSPECRAD(ls,kr)];   // j level index corresponding to transition kr

  long b_ij = LSPECLEVLEV(ls,i,j);              // frequency index


  double speed_width = sqrt(8.0*KB*cells->temperature_gas[origin]/PI/MP + pow(V_TURB, 2));

  double escape_probability = 0.0;                 // escape probability from Sobolev approximation


  // DO RADIATIVE TRANSFER
  // _ _ _ _ _ _ _ _ _ _ _


  // For half of the rays (only half is needed since we also consider antipodals)

  for (long r = 0; r < NRAYS/2; r++)
  {

    // Get antipodal ray for r

    long ar = rays.antipod[r];


    // Fill source function and optical depth increment along ray r

    double tau_r  = 0.0;
    double tau_ar = 0.0;


    // Walk along antipodal ray (ar) of r

    {
      double Z  = cells->Z[RINDEX(origin,ar)];
      double dZ = 0.0;

      long current  = cells->endpoint[RINDEX(origin,ar)];
      long previous = previous_cell (NCELLS, cells, rays, origin, ar, &Z, current, &dZ);

      long s_c = LSPECGRIDRAD(ls,current,kr);

      double chi_c = opacity[s_c];


      while ( (current != origin) && (previous != NCELLS) )
      {
        long s_p = LSPECGRIDRAD(ls,previous,kr);

        double chi_p = opacity[s_p];

        tau_ar = tau_ar + dZ*PC*(chi_c + chi_p)/2.0;

        current  = previous;
        previous = previous_cell (NCELLS, cells, rays, origin, ar, &Z, current, &dZ);

        chi_c = chi_p;
      }
    }


    // Calculate ar's contribution to escape probability

    tau_ar = CC / lines.frequency[b_ij] / speed_width * tau_ar;


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
      double Z  = 0.0;
      double dZ = 0.0;

      long current = origin;
      long next    = next_cell (NCELLS, cells, rays, origin, r, &Z, current, &dZ);

      long s_c = LSPECGRIDRAD(ls,current,kr);

      double chi_c = opacity[s_c];


      while (next != NCELLS)
      {
        long s_n = LSPECGRIDRAD(ls,next,kr);

        double chi_n = opacity[s_n];

        tau_r = tau_r + dZ*PC*(chi_c + chi_n)/2.0;

        current = next;
        next    = next_cell (NCELLS, cells, rays, origin, r, &Z, current, &dZ);

        chi_c = chi_n;
      }
    }


    // Calculate r's contribution to escape probability

    tau_r = CC / lines.frequency[b_ij] / speed_width * tau_r;


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




  // ADD CONTINUUM RADIATION (due to dust and CMB)
  // _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _


  // Continuum radiation is assumed to be local

  double factor          = 2.0*HH*pow(lines.frequency[b_ij],3)/pow(CC,2);

  double rho_grain       = 2.0;
  double ngrain          = 2.0E-12*cells->density[origin]*METALLICITY*100.0/GAS_TO_DUST;
  double emissivity_dust = rho_grain*ngrain*0.01*1.3*lines.frequency[b_ij]/3.0E11;

  double Planck_dust     = 1.0 / (exp(HH*lines.frequency[b_ij]/KB/cells->temperature_dust[origin]) - 1.0);
  double Planck_CMB      = 1.0 / (exp(HH*lines.frequency[b_ij]/KB/T_CMB) - 1.0);

  double continuum_mean_intensity = factor * (Planck_CMB + emissivity_dust*Planck_dust);


  cells->mean_intensity[KINDEX(origin,mm_ij)] = (1.0 - escape_probability) * source[m_ij]
                                               + escape_probability * continuum_mean_intensity;


  if (ACCELERATION_APPROX_LAMBDA)
  {
    Lambda_diagonal[m_ij]    = (1.0 - escape_probability);

    mean_intensity_eff[m_ij] = escape_probability * continuum_mean_intensity;
  }

  else
  {
    Lambda_diagonal[m_ij]    = 0.0;

    mean_intensity_eff[m_ij] = cells->mean_intensity[KINDEX(origin,mm_ij)];
  }


  return(0);

}
