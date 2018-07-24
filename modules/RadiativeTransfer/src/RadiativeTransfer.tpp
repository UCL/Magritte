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
		const long R  = r - START_raypair;

    const long ar = cells.rays.antipod[r];

    vReal   Su [ncells];   // effective source for u along ray r
	  vReal   Sv [ncells];   // effective source for v along ray r
	  vReal dtau [ncells];   // optical depth increment along ray r

	  vReal Lambda [ncells];


	  // Loop over all cells

#   pragma omp parallel                                                                \
	  shared  (cells, temperature, frequencies, lines, scattering, radiation, r, cout)   \
		private (Su, Sv, dtau, Lambda)                                                     \
		default (none)
    {

    const int num_threads = omp_get_num_threads();
    const int thread_num  = omp_get_thread_num();

    const long start = (thread_num*ncells)/num_threads;
    const long stop  = ((thread_num+1)*ncells)/num_threads;   // Note brackets


    for (long o = start; o < stop; o++)
	  {

	//MPI_TIMER timer_RT_CALC ("RT_CALC");
	//timer_RT_CALC.start ();
	    //MPI_TIMER timer_PS ("PS");
	    //timer_PS.start ();

			long  cellNrs_r [ncells];
			long    notch_r [ncells];
			long   lnotch_r [ncells];
		  double shifts_r [ncells];   // indicates where we are in frequency space
			double    dZs_r [ncells];

			long  cellNrs_ar [ncells];
			long    notch_ar [ncells];
			long   lnotch_ar [ncells];
		  double shifts_ar [ncells];   // indicates where we are in frequency space
			double    dZs_ar [ncells];


			// Extract the cell on ray r and antipodal ar

      long n_r  = cells.on_ray (o, r,  cellNrs_r,  dZs_r);
      long n_ar = cells.on_ray (o, ar, cellNrs_ar, dZs_ar);

	    const long ndep = n_r + n_ar;

	    if (ndep > 1)
			{

			  for (long q = 0; q < n_ar; q++)
			  {
			  	 notch_ar[q] = 0;
			  	lnotch_ar[q] = 0;
		      shifts_ar[q] = 1.0 - cells.relative_velocity (o, ar, cellNrs_ar[q]) / CC;
			  }

			  for (long q = 0; q < n_r; q++)
			  {
			  	 notch_r[q] = 0;
			  	lnotch_r[q] = 0;
		      shifts_r[q] = 1.0 - cells.relative_velocity (o, r, cellNrs_r[q]) / CC;
			  }


			  lnotch_r[ncells]  = 0;
			  lnotch_ar[ncells] = 0;


	      for (long f = 0; f < nfreq_red; f++)
	  	  {

          set_up_ray <Dimension, Nrays>
					           (cells, frequencies, temperature,lines, scattering, radiation, f, o, R,
											lnotch_ar, notch_ar, cellNrs_ar, shifts_ar, dZs_ar, n_ar,
          						lnotch_r,  notch_r,  cellNrs_r,  shifts_r,  dZs_r,  n_r,
            	        Su, Sv, dtau, ndep);


          solve_ray (ndep, Su, Sv, dtau, ndiag, Lambda, ncells);


        	vReal u_local = 0.0;   // local value of u field in direction r/ar
        	vReal v_local = 0.0;   // local value of v field in direction r/ar

	        if (n_ar > 0)
	        {
	          u_local += Su[n_ar-1];   // Su now contains u
	          v_local += Sv[n_ar-1];   // Sv now contains v
	        }

	        if (n_r > 0)
	        {
	          u_local += Su[n_ar];     // Su now contains u
	          v_local += Sv[n_ar];     // Sv now contains v
	        }

	        if ( (n_ar > 0) && (n_r > 0) )
	        {
	          u_local = 0.5*u_local;
	          v_local = 0.5*v_local;
	        }


			  	const long index = radiation.index(o,f);

	        radiation.u[R][index] = u_local;
	        radiation.v[R][index] = v_local;

	  	  } // end of loop over frequencies

	    }

			else if (ndep == 1)
			{
				// set up ray

				// trivially solve ray

			}

			else
			{
	      const long b = cells.cell_to_bdy_nr[o];

	      for (long f = 0; f < nfreq_red; f++)
	  	  {
		      const long index = radiation.index(o,f);

	        radiation.u[R][index] = 0.5 * (  radiation.boundary_intensity[r][b][f]
						                             + radiation.boundary_intensity[ar][b][f]);
	        radiation.v[R][index] = 0.5 * (  radiation.boundary_intensity[r][b][f]
						                             - radiation.boundary_intensity[ar][b][f]);
				}
		  }

			//timer_SS.stop ();
			//timer_SS.print ();

	//timer_RT_CALC.stop ();
	//timer_RT_CALC.print_to_file ();


    }
    } // end of pragma omp parallel

  } // end of loop over ray pairs



	// Reduce results of all MPI processes to get J, U and V

	MPI_TIMER timer_RT_COMM ("RT_COMM");
	timer_RT_COMM.start ();

	radiation.calc_J ();

	radiation.calc_U_and_V (scattering);

	timer_RT_COMM.stop ();
	timer_RT_COMM.print ();

	return (0);

}
