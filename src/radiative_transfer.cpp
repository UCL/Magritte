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

int radiative_transfer (long ncells, CELLS *cells, HEALPIXVECTORS healpixvectors, LINES lines,
                        double *Lambda_diagonal, double *mean_intensity_eff,
                        double *source, double *opacity, long o, int ls, int kr)
{

  long m_ij  = LSPECGRIDRAD(ls,o,kr);     // mean_intensity, S and opacity index
  long mm_ij = KINDEX(o,LSPECRAD(ls,kr));   // mean_intensity, S and opacity index

  int i = lines.irad[LSPECRAD(ls,kr)];   // i level index corresponding to transition kr
  int j = lines.jrad[LSPECRAD(ls,kr)];   // j level index corresponding to transition kr

  long b_ij = LSPECLEVLEV(ls,i,j);   // frequency index


  // For half of rays (only half is needed since we also consider antipodals)

  for (long ray = 0; ray < NRAYS/2; ray++)
  {

    // For all frequencies (Gauss-Hermite quadrature)

    for (int ny = 0; ny < NFREQ; ny++)
    {
      double u_local;
      double v_local;
      double L_local;

      double line_frequency  = lines.frequency[b_ij];

      double width = line_frequency / CC * sqrt(2.0*KB*cells->temperature_gas[o]/MP + V_TURB*V_TURB);

      double freq = line_frequency + H_4_roots[ny]*width;


      intensities (NCELLS, cells, healpixvectors, lines, source, opacity, freq,
                   o, ray, ls, kr, &u_local, &v_local, &L_local);


      cells->mean_intensity[mm_ij] = cells->mean_intensity[mm_ij] + H_4_weights[ny]/width*u_local;

      Lambda_diagonal[m_ij]        = Lambda_diagonal[m_ij]       + H_4_weights[ny]/width*L_local;

    } // end of ny loop over frequencies

  } // end of r loop over half of the rays


  cells->mean_intensity[mm_ij] = cells->mean_intensity[mm_ij] / NRAYS;


  /* Add the continuum radiation (due to dust and CMB) */

  double factor          = 2.0*HH*pow(lines.frequency[b_ij],3)/pow(CC,2);

  double rho_grain       = 2.0;

  double ngrain          = 2.0E-12*cells->density[o]*METALLICITY*100.0/GAS_TO_DUST;

  double emissivity_dust = rho_grain*ngrain*0.01*1.3*lines.frequency[b_ij]/3.0E11;

  double Planck_dust     = 1.0 / (exp(HH*lines.frequency[b_ij]/KB/cells->temperature_dust[o]) - 1.0);

  double Planck_CMB      = 1.0 / (exp(HH*lines.frequency[b_ij]/KB/T_CMB) - 1.0);


  // NOTE: Continuum radiation is assumed to be local

  double continuum_mean_intensity = factor * (Planck_CMB + emissivity_dust*Planck_dust);


  cells->mean_intensity[mm_ij] = cells->mean_intensity[mm_ij] + continuum_mean_intensity;


  if (ACCELERATION_APPROX_LAMBDA)
  {
    mean_intensity_eff[m_ij] = cells->mean_intensity[mm_ij] - Lambda_diagonal[m_ij]*source[m_ij];
  }

  else
  {
    Lambda_diagonal[m_ij] = 0.0;

    mean_intensity_eff[m_ij] = cells->mean_intensity[mm_ij];
  }


  return (0);

}




// intensity: calculate intensity along a certain ray through a certain point
// --------------------------------------------------------------------------

int intensities (long ncells, CELLS *cells, HEALPIXVECTORS healpixvectors, LINES lines, double *source, double *opacity,
                 double freq, long origin, long r, int ls, int kr,
                 double *u_local, double *v_local, double *L_local)
{

  // Get the antipodal ray for r

  long ar = healpixvectors.antipod[r];               // index of antipodal ray to r

  long bdy_ar = cells->endpoint[RINDEX(origin,ar)];   // last (boundary) cell following ar
  long bdy_r  = cells->endpoint[RINDEX(origin,r)];    // last (boundary) cell following r

  long m_ij = LSPECGRIDRAD(ls,origin,kr);            // mean_intensity, S and opacity index

  int i = lines.irad[LSPECRAD(ls,kr)];               // i level index corresponding to transition kr
  int j = lines.jrad[LSPECRAD(ls,kr)];               // j level index corresponding to transition kr

  long b_ij = LSPECLEVLEV(ls,i,j);                   // frequency index




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
    double Z  = cells->Z[RINDEX(origin,ar)];
    double dZ = 0.0;

    long current  = bdy_ar;
    long previous = previous_cell (NCELLS, cells, healpixvectors, origin, ar, &Z, current, &dZ);

    long s_c = LSPECGRIDRAD(ls,current,kr);

    double phi_c = lines.profile (NCELLS, cells, 0.0, freq, lines.frequency[b_ij], current);
    double chi_c = opacity[s_c] * phi_c;


    while ( (current != origin) && (previous != NCELLS) )
    {
      long s_p = LSPECGRIDRAD(ls,previous,kr);

      double velocity = relative_velocity (NCELLS, cells, healpixvectors, origin, ar, previous);
      double phi_p    = lines.profile (NCELLS, cells, velocity, freq, lines.frequency[b_ij], previous);
      double chi_p    = opacity[s_p] * phi_p;

      S[ndep]    = (source[s_c] + source[s_p]) / 2.0;
      dtau[ndep] = dZ * PC * (chi_c + chi_p) / 2.0;

      current  = previous;
      previous = previous_cell (NCELLS, cells, healpixvectors, origin, ar, &Z, current, &dZ);

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
    long next    = next_cell (NCELLS, cells, healpixvectors, origin, r, &Z, current, &dZ);

    long s_c = LSPECGRIDRAD(ls,current,kr);

    double phi_c = lines.profile (NCELLS, cells, 0.0, freq, lines.frequency[b_ij], current);
    double chi_c = opacity[s_c] * phi_c;


    while (next != NCELLS)
    {
      long s_n = LSPECGRIDRAD(ls,next,kr);

      double velocity = relative_velocity (NCELLS, cells, healpixvectors, origin, r, next);
      double phi_n    = lines.profile (NCELLS, cells, velocity, freq, lines.frequency[b_ij], next);
      double chi_n    = opacity[s_n] * phi_n;

      S[ndep]    = (source[s_c] + source[s_n]) / 2.0;
      dtau[ndep] = dZ * PC * (chi_c + chi_n) / 2.0;

      current = next;
      next    = next_cell (NCELLS, cells, healpixvectors, origin, r, &Z, current, &dZ);

      s_c   = s_n;
      chi_c = chi_n;

      ndep++;
    }
  }




  // Add boundary conitions
  // ______________________

  S[0]      = S[0]      + 2.0*cells->intensity[RINDEX(bdy_ar,r)]/dtau[0];
  S[ndep-1] = S[ndep-1] + 2.0*cells->intensity[RINDEX(bdy_r,ar)]/dtau[ndep-1];




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

    *v_local = cells->intensity[RINDEX(bdy_ar,r)] - u[0];

    *L_local = L_diag_approx[0];
  }

  else if (o_label == ndep)
  {
    *u_local = u[ndep-1];

    *v_local = u[ndep-1] - cells->intensity[RINDEX(bdy_r,ar)];

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
