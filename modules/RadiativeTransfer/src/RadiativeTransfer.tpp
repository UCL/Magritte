// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <vector>
using namespace std;
#include <Eigen/Core>
using namespace Eigen;

#include "declarations.hpp"
#include "RadiativeTransfer.hpp"
#include "cells.hpp"
#include "radiation.hpp"
#include "medium.hpp"
#include "set_up_ray.hpp"
#include "solve_ray.hpp"


#define RCF(r,c,f) ( Nfreq*(cells.ncells*(r) + (c)) + (f) )
#define FC(c,f) ( cells.ncells*(c) + (f) )


///  RadiativeTransfer: solves the transfer equation for the radiation field
///    @param[in] *cells: pointer to the geometric cell data containing the grid
///    @param[in/out] *radiation: pointer to (previously calculated) radiation field
///    @param[in] *medium: pointer to the opacity and emissivity data of the medium
///    @param[in] nrays: number of rays that are calculated
///    @param[in] *rays: pointer to the numbers of the rays that are calculated
///    @param[out] *J: mean intesity of the radiation field
////////////////////////////////////////////////////////////////////////////////////

template <int Dimension, long Nrays, long Nfreq>
int RadiativeTransfer (CELLS <Dimension, Nrays>& cells, RADIATION& radiation,
											 MEDIUM& medium, long nrays, long *rays, double *J)
{

  const long ndiag = 1;


	// set up opacities and sources


	long o = 0;
	long r = 0;
	long f = 0;

  long ar = cells.rays.antipod[r];   // index of antipodal ray to r


	long n_r = 0;

	std::vector <double>   Su_r (cells.ncells);    // effective source for u along ray r
	std::vector <double>   Sv_r (cells.ncells);    // effective source for v along ray ar
	std::vector <double> dtau_r (cells.ncells);    // optical depth increment along ray

	long n_ar = 0;

	std::vector <double>   Su_ar (cells.ncells);   // source function along ray
  std::vector <double>   Sv_ar (cells.ncells);   // source function along ray
  std::vector <double> dtau_ar (cells.ncells);   // optical depth increment along ray


  set_up_ray <Dimension, Nrays, Nfreq>
             (cells, radiation, medium, o, r, f,  1.0, n_r,  Su_r,  Sv_r,  dtau_r);

  set_up_ray <Dimension, Nrays, Nfreq>
             (cells, radiation, medium, o, r, f, -1.0, n_ar, Su_ar, Sv_ar, dtau_ar);

   
	const long ndep = n_r + n_ar;

  double u_loc = 0.0;
  double v_loc = 0.0;


	if (ndep > 0)
	{
		std::vector <double> u (ndep);
	  std::vector <double> v (ndep);
	
    Eigen::MatrixXd Lambda (ndep,ndep);


//    solve_ray (n_r,  Su_r,  Sv_r,  dtau_r,
//				       n_ar, Su_ar, Sv_ar, dtau_ar,
//							 ndep, Nfreq, u, v, ndiag, Lambda);


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
	  	u_loc = 0.5*u_loc;
	  	v_loc = 0.5*v_loc;
	  }
	}


	const double freq  =   radiation.frequencies[f];

	const double chi_s =   medium.chi_scat(o, freq);

	const double chi_e =   medium.chi_line(o, freq) 
		                   + medium.chi_cont(o, freq) + chi_s; 

	J[FC(o,f)] += u_loc;

	radiation.U_d[RCF(r,o,f)] += u_loc;
	radiation.V_d[RCF(r,o,f)] += v_loc;
	

	return (0);

}
