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
#include "medium.hpp"
#include "radiation.hpp"
//#include "lines.hpp"
#include "feautrier.hpp"


#define RCF(r,c,f) ( Nfreq*(cells->ncells*(r) + (c)) + (f) )
#define FC(c,f) ( cells->ncells*(c) + (f) )

const double tau_max = 1.0E9;


template <int Dimension, long Nrays, long Nfreq>
int RadiativeTransfer (CELLS <Dimension, Nrays> *cells, RADIATION *radiation,
											 MEDIUM* medium, long nrays, long *rays, double *J)
{

  const long ndiag = 1;


	// set up opacities and sources


	long o = 0;
	long r = 0;
	long f = 0;

  long ar = cells->rays.antipod[r];   // index of antipodal ray to r

	long        n_r = 0;
  double    *Su_r = new double[cells->ncells];   // effective source for u along ray r
  double    *Sv_r = new double[cells->ncells];   // effective source for v along ray ar
  double  *dtau_r = new double[cells->ncells];   // optical depth increment along ray

	long       n_ar = 0;
  double   *Su_ar = new double[cells->ncells];   // source function along ray
  double   *Sv_ar = new double[cells->ncells];   // source function along ray
  double *dtau_ar = new double[cells->ncells];   // optical depth increment along ray

	/* Find a better way to store these!!! The are almost empty!!! */

  set_up_ray <Dimension, Nrays, Nfreq>
             (cells, radiation, medium, o, r, f, 1.0, &n_r, Su_r, Sv_r, dtau_r);

  set_up_ray <Dimension, Nrays, Nfreq>
             (cells, radiation, medium, o, r, f, -1.0, &n_ar, Su_ar, Sv_ar, dtau_ar);

   
	const long ndep = n_r + n_ar;

  double u_loc = 0.0;
  double v_loc = 0.0;


	if (ndep > 0)
	{
  	double *u = new double[ndep];
	  double *v = new double[ndep];
	
    Eigen :: MatrixXd Lambda (ndep,ndep);

    feautrier (n_r, Su_r, dtau_r, n_ar, Su_ar, dtau_ar, u, Lambda, ndiag);
    feautrier (n_r, Sv_r, dtau_r, n_ar, Sv_ar, dtau_ar, v);


	  if (n_ar > 0)
	  {
	  	u_loc += u[n_ar-1];
	  	v_loc += v[n_ar-1];
	  }

	  if (n_r > 0)
	  {
	  	u_loc += u[n_ar];
	  	v_loc += v[n_ar];
	  }

	  if ( (n_ar > 0) && (n_r > 0) )
	  {
	  	u_loc = u_loc / 2.0;
	  	v_loc = v_loc / 2.0;
	  }


	  delete [] u;
	  delete [] v;
	}


	delete [] Su_r;
	delete [] Sv_r;
	delete [] dtau_r;

	delete [] Su_ar;
	delete [] Sv_ar;
	delete [] dtau_ar;

	const double freq  =   radiation->frequencies[f];

	const double chi_s =   medium->chi_scat(o, freq);

	const double chi_e =   medium->chi_line(o, freq) 
		                   + medium->chi_cont(o, freq) + chi_s; 

	J[FC(o,f)] += u_loc;

	radiation->U_d[RCF(r,o,f)] += u_loc;
	radiation->V_d[RCF(r,o,f)] += v_loc;
	

	return (0);

}




template <int Dimension, long Nrays, long Nfreq>
int set_up_ray (CELLS <Dimension, Nrays> *cells, RADIATION *radiation,
		            MEDIUM *medium, long o, long r, long f, double sign,
	              long *n, double *Su, double *Sv, double *dtau)
{

  double tau  = 0.0;   // optical depth along ray

  double  Z = 0.0;   // distance from origin (o)
  double dZ = 0.0;   // last distance increment from origin (o)

  long current = o;
  long next    = cells->next (o, r, current, &Z, &dZ);


	if (next != cells->ncells)   // if we are not going out of grid
	{
    long s_c = current; //LSPECGRIDRAD(ls,current,kr);

		const double freq = radiation->frequencies[f];

		double chi_c =   medium->chi_line (current, freq)
		               + medium->chi_cont (current, freq)
									 + medium->chi_scat (current, freq);
		
		double eta_c =   medium->eta_line (current, freq)
		               + medium->eta_cont (current, freq);

		double term1_c = (radiation->U(current, r, freq) + eta_c) / chi_c;
		double term2_c =  radiation->V(current, r, freq)          / chi_c;


		do
		{
      const double velocity = cells->relative_velocity (o, r, next);
    
			const double nu = (1.0 - velocity/CC)*freq;

			const double chi_n =   medium->chi_line (next, nu)
				                   + medium->chi_cont (next, nu)
													 + medium->chi_scat (next, nu);
			
			const double eta_n =   medium->eta_line (next, nu)
				                   + medium->eta_cont (next, nu);

			const double term1_n = (radiation->U(next, r, nu) + eta_n) / chi_n;
			const double term2_n =  radiation->V(next, r, nu)          / chi_n;

      dtau[*n] = dZ * PC * (chi_c + chi_n) / 2.0;
        Su[*n] = (term1_n + term1_c) / 2.0 + sign * (term2_n - term2_c) / dtau[*n];
        Sv[*n] = (term2_n + term2_c) / 2.0 + sign * (term1_n - term1_c) / dtau[*n];
 
			if (cells->boundary[next])
			{
				// Add boundary condition

				Su[*n] += 2.0 / dtau[*n] * (0.0 - sign * (term2_c + term2_n)/2.0);
				Sv[*n] += 2.0 / dtau[*n] * (0.0 - sign * (term1_c + term1_n)/2.0);
			}

      current = next;
      next    = cells->next (o, r, current, &Z, &dZ);
  
        chi_c =   chi_n;
      term1_c = term1_n;
      term2_c = term2_n;

			tau += dtau[*n];
      *n++;
		}
	
    while ( (!cells->boundary[current]) && (tau < tau_max) );
	}  


	return (0);

}

