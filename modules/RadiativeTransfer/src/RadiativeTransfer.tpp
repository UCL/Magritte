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
#include "radiation.hpp"
#include "medium.hpp"
#include "set_up_ray.hpp"
#include "solve_ray.hpp"


#define RCF(r,c,f) ( Nfreq*(cells->ncells*(r) + (c)) + (f) )
#define FC(c,f) ( cells->ncells*(c) + (f) )


///  RadiativeTransfer: solves the transfer equation for the radiation field
///    @param[in] *cells: pointer to the geometric cell data containing the grid
///    @param[in/out] *radiation: pointer to (previously calculated) radiation field
///    @param[in] *medium: pointer to the opacity and emissivity data of the medium
///    @param[in] nrays: number of rays that are calculated
///    @param[in] *rays: pointer to the numbers of the rays that are calculated
///    @param[out] *J: mean intesity of the radiation field
////////////////////////////////////////////////////////////////////////////////////

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


    solve_ray (n_r,  Su_r,  Sv_r,  dtau_r,
				       n_ar, Su_ar, Sv_ar, dtau_ar,
							 ndep, u, v, ndiag, Lambda);


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
