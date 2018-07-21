// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <mpi.h>
#include <omp.h>
#include <vector>
#include<iostream>
using namespace std;
#include <Eigen/Core>
using namespace Eigen;

#include "RadiativeTransfer.hpp"
#include "timer.hpp"
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
///    @param[in] temperature: data structure containing the temperature data
///    @param[in] frequencies: data structure containing the frequencies data
///    @param[in] lines: data structure containing the line transfer data
///    @param[in] scattering: data structure containing the scattering data
///    @param[in/out] radiation: reference to the  radiation field
///    @param[out] J: reference to the  radiation field
/////////////////////////////////////////////////////////////////////////////////

template <int Dimension, long Nrays>
int RadiativeTransfer (const CELLS <Dimension, Nrays>& cells, const TEMPERATURE& temperature,
		                   FREQUENCIES& frequencies, LINES& lines, const SCATTERING& scattering,
											 RADIATION& radiation)
{

  const long ndiag = 0;

	const long ncells    = cells.ncells;
	const long nfreq     = frequencies.nfreq;
	const long nfreq_red = frequencies.nfreq_red;


  int world_size;
  MPI_Comm_size (MPI_COMM_WORLD, &world_size);

  int world_rank;
  MPI_Comm_rank (MPI_COMM_WORLD, &world_rank);

  const long START_raypair = ( world_rank   *Nrays/2)/world_size;
  const long STOP_raypair  = ((world_rank+1)*Nrays/2)/world_size;


	// For all ray pairs

  for (long r = START_raypair; r < STOP_raypair; r++)
	{
		const long R = r - START_raypair;


		// Allocate all vector variables outside the omp parallel block

	  vReal eta_c;
	  vReal chi_c;

    vReal term1_c;
	  vReal term2_c;

	  vReal eta_n;
	  vReal chi_n;

	  vReal freq_scaled;

	  vReal U_scaled;
	  vReal V_scaled;

	  vReal term1_n;
	  vReal term2_n;

	  vReal u [ncells];
	  vReal v [ncells];

    vReal   Su_r [ncells];   // effective source for u along ray r
	  vReal   Sv_r [ncells];   // effective source for v along ray r
	  vReal dtau_r [ncells];   // optical depth increment along ray r

    vReal   Su_ar [ncells];   // effective source for u along ray ar
	  vReal   Sv_ar [ncells];   // effective source for v along ray ar
	  vReal dtau_ar [ncells];   // optical depth increment along ray ar

	  vReal Lambda [ncells];

    vReal A [ncells];   // A coefficient in Feautrier recursion relation
	  vReal C [ncells];   // C coefficient in Feautrier recursion relation
    vReal F [ncells];   // helper variable from Rybicki & Hummer (1991)
    vReal G [ncells];   // helper variable from Rybicki & Hummer (1991)

	  vReal B0;          // B[0]
    vReal B0_min_C0;   // B[0] - C[0]
    vReal Bd;          // B[ndep-1]
	  vReal Bd_min_Ad;   // B[ndep-1] - A[ndep-1]

    vReal u_local;   // local value of u field in direction r/ar
    vReal v_local;   // local value of v field in direction r/ar


	  // Loop over all cells

#   pragma omp parallel                                                               \
	  shared (cells, temperature, frequencies, lines, scattering, radiation, r, cout)   \
		private (eta_c, chi_c, term1_c, term2_c, eta_n, chi_n, term1_n, term2_n,          \
			       freq_scaled, U_scaled, V_scaled,                                         \
						 u, v, Su_r, Sv_r, dtau_r, Su_ar, Sv_ar, dtau_ar, Lambda, A, C, F, G,     \
		         B0, B0_min_C0, Bd, Bd_min_Ad, u_local, v_local)                          \
		default (none)                                                                    \

    {

    const int num_threads = omp_get_num_threads();
    const int thread_num  = omp_get_thread_num();

    const long start = (thread_num*ncells)/num_threads;
    const long stop  = ((thread_num+1)*ncells)/num_threads;   // Note brackets


    for (long o = start; o < stop; o++)
	  {

	//MPI_TIMER timer_RT_CALC ("RT_CALC");
	//timer_RT_CALC.start ();

			long notch = 0;

	    for (long f = 0; f < nfreq_red; f++)
	  	{

	      long n_r  = 0;
	      long n_ar = 0;

	      //MPI_TIMER timer_SU ("SU");
	      //timer_SU.start ();

        set_up_ray <Dimension, Nrays>
                   (cells, frequencies, temperature, lines, scattering, radiation,
			  					  f, notch, o, r, R,  ray,
										eta_c, chi_c, term1_c, term2_c, eta_n, chi_n, term1_n, term2_n,
								    freq_scaled, U_scaled, V_scaled, n_r,  Su_r,  Sv_r,  dtau_r);

        set_up_ray <Dimension, Nrays>
                   (cells, frequencies, temperature, lines, scattering, radiation,
										f, notch, o, r, R, antipod,
		  							eta_c, chi_c, term1_c, term2_c, eta_n, chi_n, term1_n, term2_n,
								    freq_scaled, U_scaled, V_scaled, n_ar, Su_ar, Sv_ar, dtau_ar);

			  //timer_SU.stop ();
			  //timer_SU.print ();
			//cout << "Got out" << endl;

	    const long ndep = n_r + n_ar;

//			if (cells.boundary[o])
//			{
//
//
//			  for (long d = 0; d < n_r; d++)
//			  {
//          cout <<   Su_r[d][0] << endl;
//          cout <<   Sv_r[d][0] << endl;
//          cout << dtau_r[d][0] << endl;
//			  }
//
//			  for (long d = 0; d < n_ar; d++)
//			  {
//          cout <<   Su_ar[d][0] << endl;
//          cout <<   Sv_ar[d][0] << endl;
//          cout << dtau_ar[d][0] << endl;
//			  }
//
//			}

//			if ((ndep != 9) && !(cells.boundary[o]))
//			{
// 			  cout << "ndep = " << ndep << endl;
//				cout << "o = " << o << endl;
//				cout << "r = " << r << endl;
//			}

	    //MPI_TIMER timer_SR ("SR");
	    //timer_SR.start ();

      	u_local = 0.0;
      	v_local = 0.0;

	      if (ndep > 1)
	      {

          //MatrixXd1 Lambda (frequencies.nfreq, MatrixXd (ndep, ndep));

	      	//MatrixXd temp (ndep,ndep);

	      	//for (long f = 0; f < frequencies.nfreq; f++)
	      	//{
          //  Lambda[f] = temp;
	      	//}


					//cout << "Trying to solve" << endl;

          solve_ray (n_r,  Su_r,  Sv_r,  dtau_r,
	      			       n_ar, Su_ar, Sv_ar, dtau_ar,
										 A, C, F, G,
										 B0, B0_min_C0, Bd, Bd_min_Ad,
        						 ndep, u, v, ndiag, Lambda);

					//cout << "Got solved" << endl;

	        if (n_ar > 0)
	        {
	       	  u_local += u[n_ar-1];
	       	  v_local += v[n_ar-1];
	        }

	        if (n_r > 0)
	        {
	       	  u_local += u[n_ar];
	       	  v_local += v[n_ar];
	        }

	        if ( (n_ar > 0) && (n_r > 0) )
	        {
	       	  u_local = 0.5*u_local;
	       	  v_local = 0.5*v_local;
	        }
	      }

	      radiation.u[R][radiation.index(o,f)] = u_local;
	      radiation.v[R][radiation.index(o,f)] = v_local;

			//timer_SR.stop ();
			//timer_SR.print ();

	  	} // end of loop over frequencies


	//timer_RT_CALC.stop ();
	//timer_RT_CALC.print_to_file ();


    }
    } // end of pragma omp parallel

  } // end of loop over ray pairs



	// Reduce results of all MPI processes to get J, U and V

	// MPI_TIMER timer_RT_COMM ("RT_COMM");
	// timer_RT_COMM.start ();

	radiation.calc_J ();

	radiation.calc_U_and_V (scattering);

	// timer_RT_COMM.stop ();
	// timer_RT_COMM.print_to_file ();

	return (0);

}
