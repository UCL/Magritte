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
#include "types.hpp"
#include "GridTypes.hpp"
#include "cells.hpp"
#include "lines.hpp"
#include "scattering.hpp"
#include "radiation.hpp"
#include "set_up_ray.hpp"
#include "solve_ray.hpp"


void mpi_vector_sum (vReal *in, vReal *inout, int *len, MPI_Datatype *datatype)
{
	for (int i = 0; i < *len; i++)
	{
		inout[i] = in[i] + inout[i];
	}
}


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


	radiation.initialize();


	// For all ray pairs

  for (long r = START_raypair; r < STOP_raypair; r++)
	{
		const long R = r - START_raypair;


	  // Loop over all cells

#   pragma omp parallel                                                               \
	  shared (cells, temperature, frequencies, lines, scattering, radiation, r, cout)   \
    default (none)
    {

    const int num_threads = omp_get_num_threads();
    const int thread_num  = omp_get_thread_num();

    const long start = (thread_num*ncells)/num_threads;
    const long stop  = ((thread_num+1)*ncells)/num_threads;   // Note brackets


    for (long o = start; o < stop; o++)
	  {

	    long n_r = 0;

 	    vReal2   Su_r (ncells, vReal1 (nfreq_red));   // effective source for u along ray r
	    vReal2   Sv_r (ncells, vReal1 (nfreq_red));   // effective source for v along ray r
	    vReal2 dtau_r (ncells, vReal1 (nfreq_red));   // optical depth increment along ray r

	    long n_ar = 0;

 	    vReal2   Su_ar (ncells, vReal1 (nfreq_red));   // effective source for u along ray ar
	    vReal2   Sv_ar (ncells, vReal1 (nfreq_red));   // effective source for v along ray ar
	    vReal2 dtau_ar (ncells, vReal1 (nfreq_red));   // optical depth increment along ray ar


      set_up_ray <Dimension, Nrays>
                 (cells, frequencies, temperature, lines, scattering, radiation,
									o, r, R,  ray, n_r,  Su_r,  Sv_r,  dtau_r);

      set_up_ray <Dimension, Nrays>
                 (cells, frequencies, temperature, lines, scattering, radiation,
									o, r, R, antipod, n_ar, Su_ar, Sv_ar, dtau_ar);


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


      vReal1 u_local (nfreq_red);   // local value of u field in direction r/ar
      vReal1 v_local (nfreq_red);   // local value of v field in direction r/ar


	    if (ndep > 1)
	    {
	    	vReal2 u (ndep, vReal1 (nfreq_red));
	      vReal2 v (ndep, vReal1 (nfreq_red));

        //MatrixXd1 Lambda (frequencies.nfreq, MatrixXd (ndep, ndep));

	    	//MatrixXd temp (ndep,ndep);

	    	//for (long f = 0; f < frequencies.nfreq; f++)
	    	//{
        //  Lambda[f] = temp;
	    	//}

				vReal2 Lambda (ndep, vReal1 (nfreq_red));

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
	      radiation.u[R][radiation.index(o,f)] = u_local[f];
	      radiation.v[R][radiation.index(o,f)] = v_local[f];

	      radiation.J[radiation.index(o,f)] += u_local[f];
	  	}

	  }
	  } // end of pragma omp parallel

	} // end of loop over ray pairs


  MPI_Datatype MPI_VREAL;
  MPI_Type_contiguous (n_simd_lanes, MPI_DOUBLE, &MPI_VREAL);
  MPI_Type_commit (&MPI_VREAL);


	MPI_Op MPI_VSUM;
	MPI_Op_create ( (MPI_User_function*) mpi_vector_sum, true, &MPI_VSUM);


	int ierr = MPI_Allreduce (
	             MPI_IN_PLACE,         // pointer to the data to be reduced -> here in place
	           	 radiation.J.data(),   // pointer to the data to be received
	           	 radiation.J.size(),   // size of the data to be received
	             MPI_VREAL,            // type of the reduced data
	           	 MPI_VSUM,             // reduction operation
	           	 MPI_COMM_WORLD);

	assert (ierr == 0);


	for (int w = 0; w < world_size; w++)
	{

    const long START_raypair1 = ( w   *Nrays/2)/world_size;
    const long STOP_raypair1  = ((w+1)*Nrays/2)/world_size;

    for (long r1 = START_raypair1; r1 < STOP_raypair1; r1++)
	  {
			const long R1 = r1 - START_raypair1;

	    vReal1 U_local (ncells*nfreq_red);
	    vReal1 V_local (ncells*nfreq_red);

#     pragma omp parallel                    \
	    shared (radiation, U_local, V_local)   \
      default (none)
      {

      const int nthreads = omp_get_num_threads();
      const int thread   = omp_get_thread_num();

      const long start = ( thread   *ncells)/nthreads;
      const long stop  = ((thread+1)*ncells)/nthreads;


      for (long p = start; p < stop; p++)
	    {
	      for (long f = 0; f < nfreq_red; f++)
	  	  {
	  	    U_local[radiation.index(p,f)] = 0.0;
	  	    V_local[radiation.index(p,f)] = 0.0;
	  	  }
	  	}
	  	}

    	for (long r2 = START_raypair; r2 < STOP_raypair; r2++)
	  	{
			 	const long R2 = r2 - START_raypair;

#       pragma omp parallel                                        \
	      shared (radiation, scattering, U_local, V_local, r1, r2)   \
        default (none)
        {

        const int nthreads = omp_get_num_threads();
        const int thread   = omp_get_thread_num();

        const long start = ( thread   *ncells)/nthreads;
        const long stop  = ((thread+1)*ncells)/nthreads;


        for (long p = start; p < stop; p++)
	      {
	        for (long f = 0; f < nfreq_red; f++)
	    	  {
	  		    U_local[radiation.index(p,f)] += radiation.u[R2][radiation.index(p,f)]
	  					                               * scattering.phase[r1][r2][f];
	  		    V_local[radiation.index(p,f)] += radiation.v[R2][radiation.index(p,f)]
	  					                               * scattering.phase[r1][r2][f];
	  		  }
	  		}
	    	}

	  	}


  	  int ierr_u = MPI_Reduce (
				           U_local.data(),           // pointer to the data to be reduced
  	  	           radiation.U[R1].data(),   // pointer to the data to be received
  	  	           radiation.U[R1].size(),   // size of the data to be received
  	               MPI_VREAL,                // type of the reduced data
  	  	           MPI_VSUM,                 // reduction operation
  			           w,                        // rank of root to which we reduce
  	  	           MPI_COMM_WORLD);

			assert (ierr_u == 0);


  	  int ierr_v = MPI_Reduce (
				           V_local.data(),           // pointer to the data to be reduced
  	  	           radiation.V[R1].data(),   // pointer to the data to be received
  	  	           radiation.V[R1].size(),   // size of the data to be received
  	               MPI_VREAL,                // type of the reduced data
  	  	           MPI_VSUM,                 // reduction operation
  			           w,                        // rank of root to which we reduce
  	  	           MPI_COMM_WORLD);

			assert (ierr_v == 0);
	  }
	}


	MPI_Type_free (&MPI_VREAL);
	MPI_Op_free (&MPI_VSUM);


	return (0);

}
