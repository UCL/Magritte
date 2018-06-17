// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <omp.h>
#include <vector>
#include<iostream>
using namespace std;
#include <Eigen/Core>
using namespace Eigen;

#include "RadiativeTransfer.hpp"
#include "types.hpp"
#include "GridTypes.hpp"
#include "cells.hpp"
#include "lines.hpp"
#include "scattering.hpp"
#include "radiation.hpp"
#include "set_up_ray.hpp"
#include "solve_ray.hpp"


///  RadiativeTransfer: solves the transfer equation for the radiation field
///    @param[in] cells: reference to the geometric cell data containing the grid
///    @param[in] nrays: number of rays that are calculated
///    @param[in] *rays: pointer to the numbers of the rays that are calculated
///    @param[in] lines: data structure containing the line transfer data
///    @param[in] scattering: data structure containing the scattering data
///    @param[in/out] radiation: reference to the  radiation field
/////////////////////////////////////////////////////////////////////////////////

template <int Dimension, long Nrays>
int RadiativeTransfer (const CELLS <Dimension, Nrays>& cells, const TEMPERATURE& temperature,
		                   FREQUENCIES& frequencies, const long nrays, const Long1& rays,
		                   LINES& lines, SCATTERING& scattering, RADIATION& radiation,
											 vDouble2& J)
{

  const long ndiag = 0;

	const long ncells    = cells.ncells;
	const long nfreq     = frequencies.nfreq;
	const long nfreq_red = (nfreq + n_vector_lanes - 1) / n_vector_lanes;

	frequencies.nfreq_red = nfreq_red;


  for (long ri = 0; ri < nrays/2; ri++)
	{

	  const long r  = rays[ri];                // index of ray r
    const long ar = cells.rays.antipod[r];   // index of antipodal ray to r


	  // Loop over all cells

#   pragma omp parallel                                                               \
	  shared (cells, temperature, frequencies, lines, scattering, radiation, J, cout)   \
    default (none)
    {

    const int num_threads = omp_get_num_threads();
    const int thread_num  = omp_get_thread_num();

    const long start = (thread_num*ncells)/num_threads;
    const long stop  = ((thread_num+1)*ncells)/num_threads;   // Note brackets


    for (long o = start; o < stop; o++)
	  {

	    long n_r = 0;

 	    vDouble2   Su_r (ncells, vDouble1 (nfreq_red));   // effective source for u along ray r
	    vDouble2   Sv_r (ncells, vDouble1 (nfreq_red));   // effective source for v along ray r
	    vDouble2 dtau_r (ncells, vDouble1 (nfreq_red));   // optical depth increment along ray r

	    long n_ar = 0;

 	    vDouble2   Su_ar (ncells, vDouble1 (nfreq_red));   // effective source for u along ray ar
	    vDouble2   Sv_ar (ncells, vDouble1 (nfreq_red));   // effective source for v along ray ar
	    vDouble2 dtau_ar (ncells, vDouble1 (nfreq_red));   // optical depth increment along ray ar


			cout << "r = " << r << ";  o = " << o << endl;

      set_up_ray <Dimension, Nrays>
                 (cells, frequencies, temperature, lines, scattering, radiation,
									o,  r,  1.0, n_r,  Su_r,  Sv_r,  dtau_r);

      set_up_ray <Dimension, Nrays>
                 (cells, frequencies, temperature, lines, scattering, radiation,
									o, ar, -1.0, n_ar, Su_ar, Sv_ar, dtau_ar);

			cout << "Rays are set up" << endl;

	    const long ndep = n_r + n_ar;

	//		cout << "ndep = " << ndep << endl;

	//		for (long d = 0; d < n_r; d++)
	//		{
  //      cout <<   Su_r[d][0] << endl;
  //      cout <<   Sv_r[d][0] << endl;
  //      cout << dtau_r[d][0] << endl;
	//		}

	//		for (long d = 0; d < n_ar; d++)
	//		{
  //      cout <<   Su_ar[d][0] << endl;
  //      cout <<   Sv_ar[d][0] << endl;
  //      cout << dtau_ar[d][0] << endl;
	//		}


      vDouble1 u_local (nfreq_red);   // local value of u field in direction r/ar
      vDouble1 v_local (nfreq_red);   // local value of v field in direction r/ar


	    if (ndep > 1)
	    {
	    	vDouble2 u (ndep, vDouble1 (nfreq_red));
	      vDouble2 v (ndep, vDouble1 (nfreq_red));
	    
        //MatrixXd1 Lambda (frequencies.nfreq, MatrixXd (ndep, ndep));

	    	//MatrixXd temp (ndep,ndep); 

	    	//for (long f = 0; f < frequencies.nfreq; f++)
	    	//{
        //  Lambda[f] = temp;
	    	//}
				
				vDouble2 Lambda (ndep, vDouble1 (nfreq_red));

				cout << "Just before solver..." << endl;

        solve_ray (n_r,  Su_r,  Sv_r,  dtau_r,
	    			       n_ar, Su_ar, Sv_ar, dtau_ar,
      						 ndep, nfreq_red, u, v, ndiag, Lambda);

				cout << "Just after solver!" << endl;

	      if (n_ar > 0)
	      {
	    	  for (long f = 0; f < nfreq_red; f++)
	    		{
	      	  u_local[f] += u[n_ar-1][f];
	      	  v_local[f] += v[n_ar-1][f];
	    		}
	      }

	      if (n_r > 0)
	      {
	    	  for (long f = 0; f < nfreq_red; f++)
	    		{
	      	  u_local[f] += u[n_ar][f];
	      	  v_local[f] += v[n_ar][f];
	    		}
	      }

	      if ( (n_ar > 0) && (n_r > 0) )
	      {
	    	  for (long f = 0; f < nfreq_red; f++)
	    		{
	      	  u_local[f] = 0.5*u_local[f];
	      	  v_local[f] = 0.5*v_local[f];
	    		}
	      }
	    }


	    for (long f = 0; f < nfreq_red; f++)
	  	{
	      radiation.u[r][o][f] += u_local[f];
	      radiation.v[r][o][f] += v_local[f];
	//	cout << radiation.U[r][o][f] << endl;

	      radiation.U[r][o][f] += u_local[f];
	      radiation.V[r][o][f] += v_local[f];

	      J[o][f] += u_local[f];


				//cout << u_local[f] << endl;
				//cout << v_local[f] << endl;
	  	}

	  }
	  } // end of pragma omp parallel	

	} // end of loop over (half) the rays ri


	return (0);

}
