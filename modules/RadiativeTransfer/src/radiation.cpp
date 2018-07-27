// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <mpi.h>
#include <omp.h>

#include <iostream>
#include <fstream>
using namespace std;

#include "radiation.hpp"
#include "constants.hpp"
#include "GridTypes.hpp"
#include "mpiTypes.hpp"
#include "cells.hpp"
#include "frequencies.hpp"
#include "scattering.hpp"
#include "profile.hpp"
#include "interpolation.hpp"


///  Constructor for RADIATION
//////////////////////////////

RADIATION :: RADIATION (const long num_of_cells,    const long num_of_rays,
		                    const long num_of_rays_red, const long num_of_freq_red,
												const long num_of_bdycells, const long START_raypair_input)
	: ncells        (num_of_cells)
	, nrays         (num_of_rays)
  , nrays_red     (num_of_rays_red)
  , nfreq_red     (num_of_freq_red)
	, nboundary     (num_of_bdycells)
	, START_raypair (START_raypair_input)
{


	// Size and initialize u, v, U and V

  u.resize (nrays_red);
  v.resize (nrays_red);

  U.resize (nrays_red);
  V.resize (nrays_red);

  boundary_intensity.resize (nrays_red);


	for (long r = 0; r < nrays_red; r++)
	{
	  u[r].resize (ncells*nfreq_red);
	  v[r].resize (ncells*nfreq_red);

	  U[r].resize (ncells*nfreq_red);
	  V[r].resize (ncells*nfreq_red);

	  boundary_intensity[r].resize (ncells);

		for (long p = 0; p < nboundary; p++)
		{
			boundary_intensity[r][p].resize (nfreq_red);
		}
	}

	J.resize (ncells*nfreq_red);

}   // END OF CONSTRUCTOR



long RADIATION ::
     index (const long r, const long p, const long f) const
{
	return f + (p + (r-START_raypair)*ncells)*nfreq_red;
}

long RADIATION ::
     index (const long p, const long f) const
{
	return f + p*nfreq_red;
}


int RADIATION ::
    read (const string boundary_intensity_file)
{

	return (0);

}

int RADIATION ::
    calc_boundary_intensities (const Long1& bdy_to_cell_nr,
				                       const FREQUENCIES& frequencies)
{

	for (long r = 0; r < nrays_red; r++)
	{

#   pragma omp parallel                       \
		shared (r, bdy_to_cell_nr, frequencies)   \
    default (none)
    {

    const int num_threads = omp_get_num_threads();
    const int thread_num  = omp_get_thread_num();

    const long start = ( thread_num   *nboundary)/num_threads;
    const long stop  = ((thread_num+1)*nboundary)/num_threads;


    for (long b = start; b < stop; b++)
    {
		  const long p = bdy_to_cell_nr[b];

	    for (long f = 0; f < nfreq_red; f++)
      {
				boundary_intensity[r][b][f] = Planck (T_CMB, frequencies.all[p][f]);
      }
	  }
	  } // end of pragma omp parallel
	}

	return (0);

}




void mpi_vector_sum (vReal *in, vReal *inout, int *len, MPI_Datatype *datatype)
{
	for (int i = 0; i < *len; i++)
	{
		inout[i] = in[i] + inout[i];
	}
}


int initialize (vReal1& vec)
{

# pragma omp parallel   \
	shared (vec)          \
  default (none)
  {

  const int nthreads = omp_get_num_threads();
  const int thread   = omp_get_thread_num();

  const long start = ( thread   *vec.size())/nthreads;
  const long stop  = ((thread+1)*vec.size())/nthreads;


  for (long i = start; i < stop; i++)
	{
	  vec[i] = 0.0;
	}
	} // end of pragma omp parallel


	return (0);

}



int RADIATION ::
    calc_J (void)
{

	initialize (J);


	int world_size;
  MPI_Comm_size (MPI_COMM_WORLD, &world_size);

  int world_rank;
  MPI_Comm_rank (MPI_COMM_WORLD, &world_rank);

  const long START_raypair = ( world_rank   *nrays/2)/world_size;
  const long STOP_raypair  = ((world_rank+1)*nrays/2)/world_size;

  for (long r = START_raypair; r < STOP_raypair; r++)
  {
		const long R = r - START_raypair;

#   pragma omp parallel   \
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
	      J[index(p,f)] += u[R][index(p,f)];
			}
		}
		} // end of pragma omp parallel

	} // end of r loop over rays


  MPI_Datatype MPI_VREAL;
  MPI_Type_contiguous (n_simd_lanes, MPI_DOUBLE, &MPI_VREAL);
  MPI_Type_commit (&MPI_VREAL);

	MPI_Op MPI_VSUM;
	MPI_Op_create ( (MPI_User_function*) mpi_vector_sum, true, &MPI_VSUM);


	int ierr = MPI_Allreduce (
	             MPI_IN_PLACE,      // pointer to data to be reduced -> here in place
	           	 J.data(),          // pointer to data to be received
	           	 J.size(),          // size of data to be received
	             MPI_VREAL,         // type of reduced data
	           	 MPI_VSUM,          // reduction operation
	           	 MPI_COMM_WORLD);

	assert (ierr == 0);


	MPI_Type_free (&MPI_VREAL);

	MPI_Op_free (&MPI_VSUM);


	return (0);

}




int RADIATION ::
    calc_U_and_V (const SCATTERING& scattering)

#	if (MPI_PARALLEL)

{

	vReal1 U_local (ncells*nfreq_red);
	vReal1 V_local (ncells*nfreq_red);


  int world_size;
  MPI_Comm_size (MPI_COMM_WORLD, &world_size);

  int world_rank;
  MPI_Comm_rank (MPI_COMM_WORLD, &world_rank);

  const long START_raypair = ( world_rank   *nrays/2)/world_size;
  const long STOP_raypair  = ((world_rank+1)*nrays/2)/world_size;


  MPI_Datatype MPI_VREAL;
  MPI_Type_contiguous (n_simd_lanes, MPI_DOUBLE, &MPI_VREAL);
  MPI_Type_commit (&MPI_VREAL);

  MPI_Op MPI_VSUM;
  MPI_Op_create ( (MPI_User_function*) mpi_vector_sum, true, &MPI_VSUM);


	for (int w = 0; w < world_size; w++)
	{
    const long START_raypair1 = ( w   *nrays/2)/world_size;
    const long STOP_raypair1  = ((w+1)*nrays/2)/world_size;

    for (long r1 = START_raypair1; r1 < STOP_raypair1; r1++)
	  {
			const long R1 = r1 - START_raypair1;

			initialize (U_local);
			initialize (V_local);


    	for (long r2 = START_raypair; r2 < STOP_raypair; r2++)
	  	{
			 	const long R2 = r2 - START_raypair;

#       pragma omp parallel                             \
	      shared (scattering, U_local, V_local, r1, r2)   \
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
	  		    U_local[index(p,f)] += u[R2][index(p,f)] * scattering.phase[r1][r2][f];
	  		    V_local[index(p,f)] += v[R2][index(p,f)] * scattering.phase[r1][r2][f];
	  		  }
	  		}
	    	}

	  	} // end of r2 loop over raypairs2


  	  int ierr_u = MPI_Reduce (
				           U_local.data(),    // pointer to the data to be reduced
  	  	           U[R1].data(),      // pointer to the data to be received
  	  	           ncells*nfreq_red,  // size of the data to be received
  	               MPI_VREAL,         // type of the reduced data
  	  	           MPI_VSUM,          // reduction operation
  			           w,                 // rank of root to which we reduce
  	  	           MPI_COMM_WORLD);

			assert (ierr_u == 0);


  	  int ierr_v = MPI_Reduce (
				           V_local.data(),    // pointer to the data to be reduced
  	  	           V[R1].data(),      // pointer to the data to be received
  	  	           ncells*nfreq_red,  // size of the data to be received
  	               MPI_VREAL,         // type of the reduced data
  	  	           MPI_VSUM,          // reduction operation
  			           w,                 // rank of root to which we reduce
  	  	           MPI_COMM_WORLD);

			assert (ierr_v == 0);


	  }
	}


	MPI_Type_free (&MPI_VREAL);

	MPI_Op_free (&MPI_VSUM);


	return (0);

}

#else

{

	vReal1 U_local (ncells*nfreq_red);
	vReal1 V_local (ncells*nfreq_red);

  for (long r1 = 0; r1 < nrays/2; r1++)
	{
		initialize (U_local);
		initialize (V_local);

    for (long r2 = 0; r2 < nrays/2; r2++)
	  {

#     pragma omp parallel                             \
	    shared (scattering, U_local, V_local, r1, r2)   \
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
	  	    U_local[index(p,f)] += u[r2][index(p,f)] * scattering.phase[r1][r2][f];
	  	    V_local[index(p,f)] += v[r2][index(p,f)] * scattering.phase[r1][r2][f];
	  	  }
	  	}
	    }

	  } // end of r2 loop over raypairs2

	}


	return (0);

}

#endif




int RADIATION ::
    rescale_U_and_V (FREQUENCIES& frequencies, const long p, const long R,
	  	               long& notch, vReal& freq_scaled,
	  								 vReal& U_scaled, vReal& V_scaled)

#if (GRID_SIMD)

{

	vReal nu1, nu2, U1, U2, V1, V2;

  for (int lane = 0; lane < n_simd_lanes; lane++)
	{

		double freq = freq_scaled.getlane (lane);

		search_with_notch (frequencies.all[p], notch, freq);

		const long f1    =  notch    / n_simd_lanes;
    const  int lane1 =  notch    % n_simd_lanes;

		const long f2    = (notch+1) / n_simd_lanes;
    const  int lane2 = (notch+1) % n_simd_lanes;

		//const double nu1 = frequencies.all[p][f1].getlane(lane1);
		//const double nu2 = frequencies.all[p][f2].getlane(lane2);

		//const double U1 = U[R][index(p,f1)].getlane(lane1);
		//const double U2 = U[R][index(p,f2)].getlane(lane2);

		//const double V1 = V[R][index(p,f1)].getlane(lane1);
		//const double V2 = V[R][index(p,f2)].getlane(lane2);

		//U_scaled.putlane(interpolate_linear (nu1, U1, nu2, U2, freq), lane);
		//V_scaled.putlane(interpolate_linear (nu1, V1, nu2, V2, freq), lane);

		nu1.putlane (frequencies.all[p][f1].getlane (lane1), lane);
		nu2.putlane (frequencies.all[p][f2].getlane (lane2), lane);

		 U1.putlane (U[R][index(p,f1)].getlane (lane1), lane);
		 U2.putlane (U[R][index(p,f2)].getlane (lane2), lane);

		 V1.putlane (V[R][index(p,f1)].getlane (lane1), lane);
		 V2.putlane (V[R][index(p,f2)].getlane (lane2), lane);
	}

	U_scaled = interpolate_linear (nu1, U1, nu2, U2, freq_scaled);
	V_scaled = interpolate_linear (nu1, V1, nu2, V2, freq_scaled);


 	return (0);

}

#else

{

	search_with_notch (frequencies.all[p], notch, freq_scaled);

	const long f1    = notch;
	const long f2    = notch+1;

	const double nu1 = frequencies.all[p][f1];
	const double nu2 = frequencies.all[p][f2];

	const double U1 = U[R][index(p,f1)];
	const double U2 = U[R][index(p,f2)];

	const double V1 = V[R][index(p,f1)];
	const double V2 = V[R][index(p,f2)];

	U_scaled = interpolate_linear (nu1, U1, nu2, U2, freq);
	V_scaled = interpolate_linear (nu1, V1, nu2, V2, freq);


 	return (0);

}

#endif




int RADIATION ::
    rescale_U_and_V_and_bdy_I (FREQUENCIES& frequencies, const long p, const long b,
	                             const long R, long& notch, vReal& freq_scaled,
															 vReal& U_scaled, vReal& V_scaled, vReal& Ibdy_scaled)
{

	vReal nu1, nu2, U1, U2, V1, V2, Ibdy1, Ibdy2;

  for (int lane = 0; lane < n_simd_lanes; lane++)
	{
		double freq = freq_scaled.getlane (lane);

		search_with_notch (frequencies.all[p], notch, freq);

		const long f1    =  notch    / n_simd_lanes;
    const  int lane1 =  notch    % n_simd_lanes;

		const long f2    = (notch+1) / n_simd_lanes;
    const  int lane2 = (notch+1) % n_simd_lanes;

		//const double nu1 = frequencies.all[p][f1].getlane(lane1);
		//const double nu2 = frequencies.all[p][f2].getlane(lane2);

		//const double U1 = U[R][index(p,f1)].getlane(lane1);
		//const double U2 = U[R][index(p,f2)].getlane(lane2);

		//const double V1 = V[R][index(p,f1)].getlane(lane1);
		//const double V2 = V[R][index(p,f2)].getlane(lane2);

		//const double Ibdy1 = boundary_intensity[R][b][f1].getlane(lane1);
		//const double Ibdy2 = boundary_intensity[R][b][f2].getlane(lane2);

		//   U_scaled.putlane (interpolate_linear (nu1, U1,    nu2, U2,    freq), lane);
		//   V_scaled.putlane (interpolate_linear (nu1, V1,    nu2, V2,    freq), lane);
		//Ibdy_scaled.putlane (interpolate_linear (nu1, Ibdy1, nu2, Ibdy2, freq), lane);

		  nu1.putlane (      frequencies.all[p][f1].getlane (lane1), lane);
		  nu2.putlane (      frequencies.all[p][f2].getlane (lane2), lane);

		   U1.putlane (           U[R][index(p,f1)].getlane (lane1), lane);
		   U2.putlane (           U[R][index(p,f2)].getlane (lane2), lane);

		   V1.putlane (           V[R][index(p,f1)].getlane (lane1), lane);
		   V2.putlane (           V[R][index(p,f2)].getlane (lane2), lane);

		Ibdy1.putlane (boundary_intensity[R][b][f1].getlane (lane1), lane);
		Ibdy2.putlane (boundary_intensity[R][b][f2].getlane (lane2), lane);
	}

	   U_scaled = interpolate_linear (nu1, U1,    nu2,    U2, freq_scaled);
	   V_scaled = interpolate_linear (nu1, V1,    nu2,    V2, freq_scaled);
	Ibdy_scaled = interpolate_linear (nu1, Ibdy1, nu2, Ibdy2, freq_scaled);


 	return (0);
}





#include "configure.hpp"

int RADIATION ::
    print (string OOOoutput_folder, string tag)
{

	int world_rank;
	MPI_Comm_rank (MPI_COMM_WORLD, &world_rank);


	if (world_rank == 0)
	{
		string file_name = output_folder + "J" + tag + ".txt";

    ofstream outputFile (file_name);

	  for (long p = 0; p < ncells; p++)
	  {
	  	for (int f = 0; f < nfreq_red; f++)
	  	{
#       if (GRID_SIMD)
					for (int lane = 0; lane < n_simd_lanes; lane++)
					{
	  		    outputFile << J[index(p,f)].getlane(lane) << "\t";
					}
#       else
	  		  outputFile << J[index(p,f)] << "\t";
#       endif
	  	}

	  	outputFile << endl;
	  }

	  outputFile.close ();
	}


	return (0);

}
