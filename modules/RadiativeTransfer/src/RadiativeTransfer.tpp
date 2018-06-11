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
int RadiativeTransfer (CELLS <Dimension, Nrays>& cells, TEMPERATURE& temperature,
		                   FREQUENCIES& frequencies, long nrays, long *rays,
		                   LINES& lines, SCATTERING& scattering, RADIATION& radiation,
											 vector<vector<double>>& J)
{

  const long ndiag = 0;

  for (long ri = 0; ri < nrays/2; ri++)
	{

	  long r  = rays[ri];                // index of ray r
    long ar = cells.rays.antipod[r];   // index of antipodal ray to r


	  // Loop over all cells

#   pragma omp parallel                                                                             \
	  shared (cells, temperature, frequencies, nrays, rays, lines, scattering, radiation, J, r, ar, cout)   \
    default (none)
    {

    const int num_threads = omp_get_num_threads();
    const int thread_num  = omp_get_thread_num();

    const long start = (thread_num*cells.ncells)/num_threads;
    const long stop  = ((thread_num+1)*cells.ncells)/num_threads;   // Note brackets


    for (long o = start; o < stop; o++)
	  {

	    long n_r = 0;

 	    vector<vector<double>>   Su_r (cells.ncells, vector<double> (frequencies.nfreq));    // effective source for u along ray r
	    vector<vector<double>>   Sv_r (cells.ncells, vector<double> (frequencies.nfreq));    // effective source for v along ray r
	    vector<vector<double>> dtau_r (cells.ncells, vector<double> (frequencies.nfreq));    // optical depth increment along ray r

	    long n_ar = 0;

 	    vector<vector<double>>   Su_ar (cells.ncells, vector<double> (frequencies.nfreq));    // effective source for u along ray ar
	    vector<vector<double>>   Sv_ar (cells.ncells, vector<double> (frequencies.nfreq));    // effective source for v along ray ar
	    vector<vector<double>> dtau_ar (cells.ncells, vector<double> (frequencies.nfreq));    // optical depth increment along ray ar



			cout << "r = " << r << ";  o = " << o << endl;

      set_up_ray <Dimension, Nrays>
                 (cells, frequencies, temperature, lines, scattering, radiation, o,  r,  1.0, n_r,  Su_r,  Sv_r,  dtau_r);

      set_up_ray <Dimension, Nrays>
                 (cells, frequencies, temperature, lines, scattering, radiation, o, ar, -1.0, n_ar, Su_ar, Sv_ar, dtau_ar);

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


      vector<double> u_local (frequencies.nfreq);   // local value of u field in direction r/ar
      vector<double> v_local (frequencies.nfreq);   // local value of v field in direction r/ar


	    if (ndep > 1)
	    {
	    	vector<vector<double>> u (ndep, vector<double> (frequencies.nfreq));
	      vector<vector<double>> v (ndep, vector<double> (frequencies.nfreq));
	    
        vector<MatrixXd> Lambda (frequencies.nfreq, MatrixXd (ndep, ndep));

	    	MatrixXd temp (ndep,ndep); 

	    	for (long f = 0; f < frequencies.nfreq; f++)
	    	{
          Lambda[f] = temp;
	    	}

				cout << "Just before solver..." << endl;

        solve_ray (n_r,  Su_r,  Sv_r,  dtau_r,
	    			       n_ar, Su_ar, Sv_ar, dtau_ar,
      						 ndep, frequencies.nfreq, u, v, ndiag, Lambda);

				cout << "Just after solver!" << endl;

	      if (n_ar > 0)
	      {
	    	  for (long f = 0; f < frequencies.nfreq; f++)
	    		{
	      	  u_local[f] += u[n_ar-1][f];
	      	  v_local[f] += v[n_ar-1][f];
	    		}
	      }

	      if (n_r > 0)
	      {
	    	  for (long f = 0; f < frequencies.nfreq; f++)
	    		{
	      	  u_local[f] += u[n_ar][f];
	      	  v_local[f] += v[n_ar][f];
	    		}
	      }

	      if ( (n_ar > 0) && (n_r > 0) )
	      {
	    	  for (long f = 0; f < frequencies.nfreq; f++)
	    		{
	      	  u_local[f] = 0.5*u_local[f];
	      	  v_local[f] = 0.5*v_local[f];
	    		}
	      }
	    }


	    for (long f = 0; f < frequencies.nfreq; f++)
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
