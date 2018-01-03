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

#if (CELL_BASED)

#include "cell_radiative_transfer.hpp"
#include "ray_tracing.hpp"
#include "lines.hpp"
#include "cell_feautrier.hpp"


// cell_radiative_transfer: calculate mean intensity at a cell
// -----------------------------------------------------------

int cell_radiative_transfer( CELL *cell, double *mean_intensity, double *Lambda_diagonal,
                             double *mean_intensity_eff, double *source, double *opacity,
                             double *frequency, double *temperature_gas, double *temperature_dust,
                             int *irad, int*jrad, long gridp, int lspec, int kr )
{


  long m_ij = LSPECGRIDRAD(lspec,gridp,kr);   // mean_intensity, S and opacity index

  int i = irad[LSPECRAD(lspec,kr)];           // i level index corresponding to transition kr
  int j = jrad[LSPECRAD(lspec,kr)];           // j level index corresponding to transition kr

  long b_ij = LSPECLEVLEV(lspec,i,j);         // frequency index


  // For half of rays (only half is needed since we also consider antipodals)

  for (long ray = 0; ray < NRAYS/2; ray++)
  {

    /* For all frequencies (Gauss-Hermite quadrature) */

    for (int ny = 0; ny < NFREQ; ny++)
    {
      double u_local;
      double v_local;
      double L_local;

      double line_frequency  = frequency[b_ij];

      double width = line_frequency / CC * sqrt(2.0*KB*cell[gridp].temperature.gas/MP + V_TURB*V_TURB);

      double freq = H_4_roots[ny]*width;


      intensities (cell, source, opacity, frequency, freq, temperature_gas, irad, jrad,
                   gridp, ray, lspec, kr, &u_local, &v_local, &L_local);


      mean_intensity[m_ij]  = mean_intensity[m_ij]  + H_4_weights[ny]/width*u_local;

      Lambda_diagonal[m_ij] = Lambda_diagonal[m_ij] + H_4_weights[ny]/width*L_local;

    } // end of ny loop over frequencies

  } // end of r loop over half of the rays


  mean_intensity[m_ij] = mean_intensity[m_ij] / NRAYS;


  /* Add the continuum radiation (due to dust and CMB) */

  double factor          = 2.0*HH*pow(frequency[b_ij],3)/pow(CC,2);

  double rho_grain       = 2.0;

  double ngrain          = 2.0E-12*cell[gridp].density*METALLICITY*100.0/GAS_TO_DUST;

  double emissivity_dust = rho_grain*ngrain*0.01*1.3*frequency[b_ij]/3.0E11;

  double Planck_dust     = 1.0 / (exp(HH*frequency[b_ij]/KB/cell[gridp].temperature.dust) - 1.0);

  double Planck_CMB      = 1.0 / (exp(HH*frequency[b_ij]/KB/T_CMB) - 1.0);


  /* NOTE: Continuum radiation is assumed to be local */

  double continuum_mean_intensity = factor * (Planck_CMB + emissivity_dust*Planck_dust);


  mean_intensity[m_ij] = mean_intensity[m_ij] + continuum_mean_intensity;


  if (ACCELERATION_APPROX_LAMBDA)
  {
    mean_intensity_eff[m_ij] = mean_intensity[m_ij] - Lambda_diagonal[m_ij]*source[m_ij];
  }

  else
  {
    Lambda_diagonal[m_ij] = 0.0;

    mean_intensity_eff[m_ij] = mean_intensity[m_ij];
  }


  return(0);

}




// intensity: calculate intensity along a certain ray through a certain point
// --------------------------------------------------------------------------

int intensities( CELL *cell, double *source, double *opacity, double *frequency,
                 double freq, double *temperature_gas,  int *irad, int*jrad, long origin, long r,
                 int lspec, int kr, double *u_local, double *v_local, double *L_local )
{

  // Get the antipodal ray for r

  long ar = antipod[r];                       // index of antipodal ray to r

  long m_ij = LSPECGRIDRAD(lspec,origin,kr);   // mean_intensity, S and opacity index

  int i = irad[LSPECRAD(lspec,kr)];           // i level index corresponding to transition kr
  int j = jrad[LSPECRAD(lspec,kr)];           // j level index corresponding to transition kr

  long b_ij = LSPECLEVLEV(lspec,i,j);         // frequency index




  // Fill source function and optical depth increment on subgrid
  //____________________________________________________________


  long ndep = 0;

  double tau_r  = 0.0;
  double tau_ar = 0.0;

  double S[NCELLS];               // source function along this ray
  double dtau[NCELLS];            // optical depth increment along this ray

  double u[NCELLS];               // Feautrier's mean intensity
  double L_diag_approx[NCELLS];   // Diagonal elements of Lambda operator


  // Walk along antipodal ray (ar) of r

  {
    double Z    = 0.0;
    double dZ   = 0.0;

    long current = origin;
    long next    = next_cell(NCELLS, cell, origin, ar, Z, current, &dZ);


    while ( (next != NCELLS) && (tau_ar < TAU_MAX) )
    {
      long s_n = LSPECGRIDRAD(lspec,next,kr);
      long s_c = LSPECGRIDRAD(lspec,current,kr);

      double velocity = relative_velocity(NCELLS, cell, origin, ar, current);

      double phi_n = cell_line_profile (NCELLS, cell, velocity, freq, frequency[b_ij], next);
      double phi_c = cell_line_profile (NCELLS, cell, velocity, freq, frequency[b_ij], current);

      S[ndep]    = (source[s_n] + source[s_c]) / 2.0;
      dtau[ndep] = dZ * PC * (opacity[s_n]*phi_n + opacity[s_c]*phi_c) / 2.0;

      tau_ar = tau_ar + dtau[ndep];
      Z      = Z + dZ;

      current = next;
      next    = next_cell(NCELLS, cell, origin, ar, Z, current, &dZ);

      ndep++;
    }
  }


  long o_label = ndep;


  // Walk along r itself

  {
    double Z    = 0.0;
    double dZ   = 0.0;

    long current = origin;
    long next    = next_cell(NCELLS, cell, origin, r, Z, current, &dZ);


    while ( (next != NCELLS) && (tau_r < TAU_MAX) )
    {
      long s_n = LSPECGRIDRAD(lspec,next,kr);
      long s_c = LSPECGRIDRAD(lspec,current,kr);

      double velocity = relative_velocity(NCELLS, cell, origin, r, current);

      double phi_n = cell_line_profile (NCELLS, cell, velocity, freq, frequency[b_ij], next);
      double phi_c = cell_line_profile (NCELLS, cell, velocity, freq, frequency[b_ij], current);

      S[ndep]    = (source[s_n] + source[s_c]) / 2.0;
      dtau[ndep] = dZ * PC * (opacity[s_n]*phi_n + opacity[s_c]*phi_c) / 2.0;

      tau_ar = tau_ar + dtau[ndep];
      Z      = Z + dZ;

      current = next;
      next    = next_cell(NCELLS, cell, origin, r, Z, current, &dZ);

      ndep++;
    }
  }




  // Avoid too small optical depth increments
  // ________________________________________

  for (long dep = 0; dep < ndep; dep++)
  {
    if (dtau[dep] < 1.0E-99)
    {
      dtau[dep] = 1.0E-99;
    }

  }




  // Solve transfer equation with Feautrier solver (on subgrid)
  // ____________________________ _____________________________


  cell_feautrier (ndep, origin, r, S, dtau, u, L_diag_approx);




  // Map results back from subgrid to grid
  // _____________________________________


  if (o_label == 0)
  {
    *u_local = u[o_label];

    *v_local = 0.0;   // No meaning since it is a boundary condition

    *L_local = L_diag_approx[o_label];
  }

  else if (o_label == ndep)
  {
    *u_local = u[o_label-1];

    *v_local = 0.0;   // No meaning since it is a boundary condition

    *L_local = L_diag_approx[o_label-1];
  }

  else
  {
    *u_local = (u[o_label] + u[o_label-1]) / 2.0;

    *v_local = 2.0 * (u[o_label] - u[o_label-1]) / (dtau[o_label] + dtau[o_label-1]) ;

    *L_local = (L_diag_approx[o_label] + L_diag_approx[o_label-1]) / 2.0;
  }


  return(0);

}


#endif
