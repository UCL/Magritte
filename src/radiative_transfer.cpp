// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>

#include "declarations.hpp"
#include "radiative_transfer.hpp"
#include "ray_tracing.hpp"
#include "lines.hpp"
#include "feautrier.hpp"


// radiative_transfer: calculate mean intensity at a cell
// -----------------------------------------------------------

int radiative_transfer (long ncells, CELL *cell, HEALPIXVECTORS healpixvectors, LINE_SPECIES line_species,
                        double *Lambda_diagonal, double *mean_intensity_eff,
                        double *source, double *opacity, long o, int lspec, int kr)
{

  long m_ij  = LSPECGRIDRAD(lspec,o,kr);   // mean_intensity, S and opacity index
  long mm_ij = LSPECRAD(lspec,kr);         // mean_intensity, S and opacity index

  int i = line_species.irad[mm_ij];   // i level index corresponding to transition kr
  int j = line_species.jrad[mm_ij];   // j level index corresponding to transition kr

  long b_ij = LSPECLEVLEV(lspec,i,j);   // frequency index


  // For half of rays (only half is needed since we also consider antipodals)

  for (long ray = 0; ray < NRAYS/2; ray++)
  {

    // For all frequencies (Gauss-Hermite quadrature)

    for (int ny = 0; ny < NFREQ; ny++)
    {
      double u_local;
      double v_local;
      double L_local;

      double line_frequency  = line_species.frequency[b_ij];

      double width = line_frequency / CC * sqrt(2.0*KB*cell[o].temperature.gas/MP + V_TURB*V_TURB);

      double freq = line_frequency + H_4_roots[ny]*width;


      intensities (NCELLS, cell, healpixvectors, line_species, source, opacity, freq,
                   o, ray, lspec, kr, &u_local, &v_local, &L_local);


      cell[o].mean_intensity[mm_ij] = cell[o].mean_intensity[mm_ij]  + H_4_weights[ny]/width*u_local;

      Lambda_diagonal[m_ij]         = Lambda_diagonal[m_ij] + H_4_weights[ny]/width*L_local;

    } // end of ny loop over frequencies

  } // end of r loop over half of the rays


  cell[o].mean_intensity[mm_ij] = cell[o].mean_intensity[mm_ij] / NRAYS;


  /* Add the continuum radiation (due to dust and CMB) */

  double factor          = 2.0*HH*pow(line_species.frequency[b_ij],3)/pow(CC,2);

  double rho_grain       = 2.0;

  double ngrain          = 2.0E-12*cell[o].density*METALLICITY*100.0/GAS_TO_DUST;

  double emissivity_dust = rho_grain*ngrain*0.01*1.3*line_species.frequency[b_ij]/3.0E11;

  double Planck_dust     = 1.0 / (exp(HH*line_species.frequency[b_ij]/KB/cell[o].temperature.dust) - 1.0);

  double Planck_CMB      = 1.0 / (exp(HH*line_species.frequency[b_ij]/KB/T_CMB) - 1.0);


  /* NOTE: Continuum radiation is assumed to be local */

  double continuum_mean_intensity = factor * (Planck_CMB + emissivity_dust*Planck_dust);


  cell[o].mean_intensity[mm_ij] = cell[o].mean_intensity[mm_ij] + continuum_mean_intensity;


  if (ACCELERATION_APPROX_LAMBDA)
  {
    mean_intensity_eff[m_ij] = cell[o].mean_intensity[mm_ij] - Lambda_diagonal[m_ij]*source[m_ij];
  }

  else
  {
    Lambda_diagonal[m_ij] = 0.0;

    mean_intensity_eff[m_ij] = cell[o].mean_intensity[mm_ij];
  }


  return (0);

}




// intensity: calculate intensity along a certain ray through a certain point
// --------------------------------------------------------------------------

int intensities (long ncells, CELL *cell, HEALPIXVECTORS healpixvectors, LINE_SPECIES line_species, double *source, double *opacity,
                 double freq, long origin, long r, int lspec, int kr,
                 double *u_local, double *v_local, double *L_local)
{

  // Get the antipodal ray for r

  long ar = healpixvectors.antipod[r];             // index of antipodal ray to r

  long bdy_ar = cell[origin].endpoint[ar];
  long bdy_r  = cell[origin].endpoint[r];

  long m_ij = LSPECGRIDRAD(lspec,origin,kr);       // mean_intensity, S and opacity index

  int i = line_species.irad[LSPECRAD(lspec,kr)];   // i level index corresponding to transition kr
  int j = line_species.jrad[LSPECRAD(lspec,kr)];   // j level index corresponding to transition kr

  long b_ij = LSPECLEVLEV(lspec,i,j);              // frequency index




  // Fill source function and optical depth increment on subgrid
  //____________________________________________________________


  long ndep = 0;

  double tau_r  = 0.0;
  double tau_ar = 0.0;

  double S[NCELLS];               // source function along this ray
  double dtau[NCELLS];            // optical depth increment along this ray

  double u[NCELLS];               // Feautrier's mean intensity
  double L_diag_approx[NCELLS];   // Diagonal elements of Lambda operator


  // Walk back along antipodal ray (ar) of r

  {
    double Z  = cell[origin].Z[ar];
    double dZ = 0.0;

    long current  = bdy_ar;
    long previous = previous_cell (NCELLS, cell, healpixvectors, origin, ar, &Z, current, &dZ);

    long s_c = LSPECGRIDRAD(lspec,current,kr);

    double phi_c = line_profile (NCELLS, cell, 0.0, freq, line_species.frequency[b_ij], current);
    double chi_c = opacity[s_c] * phi_c;


    while ( (current != origin) && (previous != NCELLS) )
    {
      long s_p = LSPECGRIDRAD(lspec,previous,kr);

      double velocity = relative_velocity (NCELLS, cell, healpixvectors, origin, ar, previous);
      double phi_p    = line_profile (NCELLS, cell, velocity, freq, line_species.frequency[b_ij], previous);
      double chi_p    = opacity[s_p] * phi_p;

      S[ndep]    = (source[s_c] + source[s_p]) / 2.0;
      dtau[ndep] = dZ * PC * (chi_c + chi_p) / 2.0;

      current  = previous;
      previous = previous_cell (NCELLS, cell, healpixvectors, origin, ar, &Z, current, &dZ);

      s_c   = s_p;
      chi_c = chi_p;

      ndep++;
    }
  }


  long o_label = ndep;


  // Walk along r itself

  {
    double Z  = 0.0;
    double dZ = 0.0;

    long current = origin;
    long next    = next_cell (NCELLS, cell, healpixvectors, origin, r, &Z, current, &dZ);

    long s_c = LSPECGRIDRAD(lspec,current,kr);

    double phi_c = line_profile (NCELLS, cell, 0.0, freq, line_species.frequency[b_ij], current);
    double chi_c = opacity[s_c] * phi_c;


    while (next != NCELLS)
    {
      long s_n = LSPECGRIDRAD(lspec,next,kr);

      double velocity = relative_velocity (NCELLS, cell, healpixvectors, origin, r, next);
      double phi_n    = line_profile (NCELLS, cell, velocity, freq, line_species.frequency[b_ij], next);
      double chi_n    = opacity[s_n] * phi_n;

      S[ndep]    = (source[s_c] + source[s_n]) / 2.0;
      dtau[ndep] = dZ * PC * (chi_c + chi_n) / 2.0;

      current = next;
      next    = next_cell (NCELLS, cell, healpixvectors, origin, r, &Z, current, &dZ);

      s_c   = s_n;
      chi_c = chi_n;

      ndep++;
    }
  }




  // Add boundary conitions
  // ______________________

  S[0]      = S[0]      + 2.0*cell[bdy_ar].ray[r].intensity/dtau[0];
  S[ndep-1] = S[ndep-1] + 2.0*cell[bdy_r].ray[ar].intensity/dtau[ndep-1];




  // Solve transfer equation with Feautrier solver (on subgrid)
  // __________________________________________________________

  if (ndep > 0)
  {
    feautrier (ndep, origin, r, S, dtau, u, L_diag_approx);
  }

  else
  {
    u[0] = 0.0;
    L_diag_approx[0] = 0.0;
  }



  // Map results back from subgrid to grid
  // _____________________________________


  if (o_label == 0)
  {
    *u_local = u[0];

    *v_local = cell[bdy_ar].ray[r].intensity - u[0];

    *L_local = L_diag_approx[0];
  }

  else if (o_label == ndep)
  {
    *u_local = u[ndep-1];

    *v_local = u[ndep-1] - cell[bdy_r].ray[ar].intensity;

    *L_local = L_diag_approx[ndep-1];
  }

  else
  {
    *u_local = (u[o_label] + u[o_label-1]) / 2.0;

    *v_local = 2.0 * (u[o_label] - u[o_label-1]) / (dtau[o_label] + dtau[o_label-1]) ;

    *L_local = (L_diag_approx[o_label] + L_diag_approx[o_label-1]) / 2.0;
  }


  return (0);

}
