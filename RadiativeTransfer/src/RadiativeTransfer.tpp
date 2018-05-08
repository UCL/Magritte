// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <math.h>
#include <iostream>
#include <Eigen/Core>

#include "declarations.hpp"
#include "RadiativeTransfer.hpp"
#include "cells.hpp"
//#include "lines.hpp"
#include "feautrier.hpp"


template <int Dimension, long Nrays, long Nfreq>
int RadiativeTransfer (CELLS <Dimension, Nrays> *cells, long *freq, long *rays,
											 double *source, double *opacity)
{

	long o = 0;
	long r = 0;

  long ar = cells->rays.antipod[r];   // index of antipodal ray to r

  double     *S_r = new double[cells->ncells];   // source function along ray
  double  *dtau_r = new double[cells->ncells];   // optical depth increment along ray

  double    *S_ar = new double[cells->ncells];   // source function along ray
  double *dtau_ar = new double[cells->ncells];   // optical depth increment along ray


  long n_r = set_up_ray <Dimension, Nrays, Nfreq>
                        (cells, o, r, source, opacity, S_r, dtau_r);

  long n_ar = set_up_ray <Dimension, Nrays, Nfreq>
                         (cells, o, ar, source, opacity, S_ar, dtau_ar);
//  solve_transfer_equation ();  
	
	std::cout << "test" << std::endl;
	std::cout << cells->x[0] << std::endl;

	return (0);

}




template <int Dimension, long Nrays, long Nfreq>
long set_up_ray (CELLS <Dimension, Nrays> *cells, long o, long r,
	    				   double *source, double *opacity,
	               double *S, double *dtau)
{

	long      n = 0;     // index along ray
  double tau  = 0.0;   // optical depth along ray

  double  Z = 0.0;
  double dZ = 0.0;

  long current = o;
  long next    = cells->next (o, r, current, &Z, &dZ);

	if (next != cells->ncells)   // if we are not going out of grid
	{
    long s_c = current; //LSPECGRIDRAD(ls,current,kr);

//  double phi_c = lines.profile (NCELLS, cells, 0.0, freq, lines.frequency[b_ij], current);
    double chi_c = opacity[s_c]; // * phi_c;

		do
		{
      long s_n = next; //LSPECGRIDRAD(ls,next,kr);
  
      double velocity = cells->relative_velocity (o, r, next);
  //    double phi_n    = lines.profile (NCELLS, cells, velocity, freq, lines.frequency[b_ij], next);
      double chi_n    = opacity[s_n]; // * phi_n;
  
         S[n] = (source[s_c] + source[s_n]) / 2.0;
      dtau[n] = dZ * PC * (chi_c + chi_n) / 2.0;
  
      current = next;
      next    = cells->next (o, r, current, &Z, &dZ);
  
        s_c =   s_n;
      chi_c = chi_n;
  
      n++;
    }
    while (!cells[next].boundary);
	}  


  // Add boundary conitions

//  S[n-1] = S[n-1] + 2.0*cells->intensity[RINDEX(bdy_r,ar)]/dtau[ndep-1];

	return n;

}

// radiative_transfer: calculate mean intensity at a cell
// -----------------------------------------------------------
//
//int radiative_transfer (CELLS *cells, RAYS rays, LINES lines,
//                        double *Lambda_diagonal, double *mean_intensity_eff,
//                        double *source, double *opacity, long o, int ls, int kr)
//{
//
//  long m_ij  = LSPECGRIDRAD(ls,o,kr);     // mean_intensity, S and opacity index
//  long mm_ij = KINDEX(o,LSPECRAD(ls,kr));   // mean_intensity, S and opacity index
//
//  int i = lines.irad[LSPECRAD(ls,kr)];   // i level index corresponding to transition kr
//  int j = lines.jrad[LSPECRAD(ls,kr)];   // j level index corresponding to transition kr
//
//  long b_ij = LSPECLEVLEV(ls,i,j);   // frequency index
//
//
//  // For half of rays (only half is needed since we also consider antipodals)
//
//  for (long ray = 0; ray < NRAYS/2; ray++)
//  {
//
//    // For all frequencies (Gauss-Hermite quadrature)
//
//    for (int ny = 0; ny < NFREQ; ny++)
//    {
//      double u_local;
//      double v_local;
//      double L_local;
//
//      double line_frequency  = lines.frequency[b_ij];
//
//      double width = line_frequency / CC * sqrt(2.0*KB*cells->temperature_gas[o]/MP + V_TURB*V_TURB);
//
//      double freq = line_frequency + H_4_roots[ny]*width;
//
//
//      intensities (NCELLS, cells, rays, lines, source, opacity, freq,
//                   o, ray, ls, kr, &u_local, &v_local, &L_local);
//
//
//      cells->mean_intensity[mm_ij] = cells->mean_intensity[mm_ij] + H_4_weights[ny]/width*u_local;
//
//      Lambda_diagonal[m_ij]        = Lambda_diagonal[m_ij]       + H_4_weights[ny]/width*L_local;
//
//    } // end of ny loop over frequencies
//
//  } // end of r loop over half of the rays
//
//
//  cells->mean_intensity[mm_ij] = cells->mean_intensity[mm_ij] / NRAYS;
//
//
//  /* Add the continuum radiation (due to dust and CMB) */
//
//  double factor          = 2.0*HH*pow(lines.frequency[b_ij],3)/pow(CC,2);
//
//  double rho_grain       = 2.0;
//
//  double ngrain          = 2.0E-12*cells->density[o]*METALLICITY*100.0/GAS_TO_DUST;
//
//  double emissivity_dust = rho_grain*ngrain*0.01*1.3*lines.frequency[b_ij]/3.0E11;
//
//  double Planck_dust     = 1.0 / expm1(HH*lines.frequency[b_ij]/KB/cells->temperature_dust[o]);
//
//  double Planck_CMB      = 1.0 / expm1(HH*lines.frequency[b_ij]/KB/T_CMB);
//
//
//  // NOTE: Continuum radiation is assumed to be local
//
//  double continuum_mean_intensity = factor * (Planck_CMB + emissivity_dust*Planck_dust);
//
//
//  cells->mean_intensity[mm_ij] = cells->mean_intensity[mm_ij] + continuum_mean_intensity;
//
//
//  if (ACCELERATION_APPROX_LAMBDA)
//  {
//    mean_intensity_eff[m_ij] = cells->mean_intensity[mm_ij] - Lambda_diagonal[m_ij]*source[m_ij];
//  }
//
//  else
//  {
//    Lambda_diagonal[m_ij] = 0.0;
//
//    mean_intensity_eff[m_ij] = cells->mean_intensity[mm_ij];
//  }
//
//
//  return (0);
//
//}
//

//
/////  solve_transfer: solve transfer equation for a certain cell along a certain ray
/////    @param[in] *cells: pointer to cell data of the whole grid
/////    @param[in] *source: pointer to source data of the whole grid
/////    @param[in] *opacity: pointer to opacity data of the whole grid
/////    @param[in] o: number of cell for which transfer eq. is solved
/////    @param[in] r: nuber or ray along which transfer eq. is solved
/////    @param[out] u_local: Feautrier u field in cell "o"
/////    @param[out] v_local: Feautrier v field in cell "o"
/////    @param[out] L_local: approximated Lambda operator in cell "o"
/////////////////////////////////////////////////////////////////////////////////////
//
//int solve_transfer (double *source, double *opacity, long o, long r,
//		                double *u_local, Eigen::Ref <Eigen::MatrixXd> Lambda)
//{
//
//  long ar = cells->rays.antipod[r];   // index of antipodal ray to r
//
//
//  // Fill source function and optical depth increment on subgrid
//  //____________________________________________________________
//
//
//  long n_r  = 0;            // index along ray r
//  long n_ar = 0;            // index along ray ar
//
//  double tau_r  = 0.0;      // optical depth along ray r
//  double tau_ar = 0.0;      // optical depth along ray ar
//
//  double     S_r[NCELLS];   // source function along ray r
//  double  dtau_r[NCELLS];   // optical depth increment along ray r
//
//  double    S_ar[NCELLS];   // source function along ray ar
//  double dtau_ar[NCELLS];   // optical depth increment along ray ar
//
//	// Can be put in only two arrays if that would help...
//	
//
//
//  // Walk along r
//
//  {
//    double Z  = 0.0;
//    double dZ = 0.0;
//
//    long current = o;
////    long next    = cells->next (o, r, current, &Z, &dZ);
//
////    long s_c = LSPECGRIDRAD(ls,current,kr);
//
////    double phi_c = lines.profile (NCELLS, cells, 0.0, freq, lines.frequency[b_ij], current);
//    double chi_c = opacity[s_c] * phi_c;
//
//
//		do
//    {
//      long s_n = LSPECGRIDRAD(ls,next,kr);
//
//      double velocity = cells->relative_velocity (o, r, next);
////      double phi_n    = lines.profile (NCELLS, cells, velocity, freq, lines.frequency[b_ij], next);
//      double chi_n    = opacity[s_n] * phi_n;
//
//         S_r[n_r] = (source[s_c] + source[s_n]) / 2.0;
//      dtau_r[n_r] = dZ * PC * (chi_c + chi_n) / 2.0;
//
//      current = next;
//      next    = cells->next (o, r, current, &Z, &dZ);
//
//      s_c   = s_n;
//      chi_c = chi_n;
//
//      n_r++;
//    }
//    while (!cells[next].boundary)
//  }
//
//
//
//
//  // Add boundary conitions
//  // ______________________
//
//  S_ar[n_ar] = S_ar[n_ar] + 2.0*cells->intensity[RINDEX(bdy_ar,r)]/dtau[0];
//  S_r[n_r]   = S_r[n_r]   + 2.0*cells->intensity[RINDEX(bdy_r,ar)]/dtau[ndep-1];
//
//
//
//
//  // Solve transfer equation with Feautrier solver (on subgrid)
//  // __________________________________________________________
//
//  if ( (n_ar > 0) || (n_r > 0) )
//  {
////    feautrier (ndep, o, r, S, dtau, u, L_diag_approx);
//  }
//
//  else
//  {
//    u[0] = 0.0;
////    L_diag_approx[0] = 0.0;
//  }
//
//
//
//  // Map results back from subgrid to grid
//  // _____________________________________
//
//
//  if (o_label == 0)
//  {
//    *u_local = u[0];
//
//    *v_local = cells->intensity[RINDEX(bdy_ar,r)] - u[0];
//
//    *L_local = L_diag_approx[0];
//  }
//
//  else if (o_label == ndep)
//  {
//    *u_local = u[ndep-1];
//
//    *v_local = u[ndep-1] - cells->intensity[RINDEX(bdy_r,ar)];
//
//    *L_local = L_diag_approx[ndep-1];
//  }
//
//  else
//  {
//    *u_local = (u[o_label] + u[o_label-1]) / 2.0;
//
//    *v_local = 2.0 * (u[o_label] - u[o_label-1]) / (dtau[o_label] + dtau[o_label-1]) ;
//
//    *L_local = (L_diag_approx[o_label] + L_diag_approx[o_label-1]) / 2.0;
//  }
//
//
//  return (0);
//
//}
