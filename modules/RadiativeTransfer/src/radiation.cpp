// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <mpi.h>
#include <omp.h>

#include <iostream>
#include <fstream>
#include <iomanip>
using namespace std;

#include "radiation.hpp"
#include "folders.hpp"
#include "constants.hpp"
#include "GridTypes.hpp"
#include "mpiTools.hpp"
#include "ompTools.hpp"
#include "frequencies.hpp"
#include "scattering.hpp"
#include "profile.hpp"
#include "interpolation.hpp"


///  Constructor for RADIATION
//////////////////////////////

RADIATION ::
RADIATION (const long num_of_cells,
           const long num_of_rays,
           const long num_of_freq_red,
           const long num_of_bdycells )
  : ncells        (num_of_cells)
  , nrays         (num_of_rays)
  , nrays_red     (get_nrays_red (nrays))
  , nfreq_red     (num_of_freq_red)
  , nboundary     (num_of_bdycells)
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




///  get_nrays_red: get reduced number of rays
///    @param[in] nrays: total number or rays
//////////////////////////////////////////////

long RADIATION :: get_nrays_red (const long nrays)
{

  int world_size;
  MPI_Comm_size (MPI_COMM_WORLD, &world_size);

  int world_rank;
  MPI_Comm_rank (MPI_COMM_WORLD, &world_rank);

  const long START_raypair = ( world_rank   *nrays/2)/world_size;
  const long STOP_raypair  = ((world_rank+1)*nrays/2)/world_size;


  return STOP_raypair - START_raypair;
  
}




///  read: read radiation field from file
/////////////////////////////////////////

int RADIATION ::
    read (const string boundary_intensity_file)
{

  return (0);

}




///  calc_boundary_intensities: calculate the boundary intensities
//////////////////////////////////////////////////////////////////

int RADIATION ::
    calc_boundary_intensities (const Long1       &bdy_to_cell_nr,
                               const FREQUENCIES &frequencies    )
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
        boundary_intensity[r][b][f] = planck (T_CMB, frequencies.nu[p][f]);
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
        J[index(p,f)] += (2.0/nrays) * u[R][index(p,f)];
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

#if (MPI_PARALLEL)

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
    print (const string tag) const
{

  int world_rank;
  MPI_Comm_rank (MPI_COMM_WORLD, &world_rank);


  if (world_rank == 0)
  {
    const string file_name_J = output_folder + "J" + tag + ".txt";

    ofstream outputFile_J (file_name_J);

    outputFile_J << scientific << setprecision(16);


    for (long p = 0; p < ncells; p++)
    {
      for (int f = 0; f < nfreq_red; f++)
      {
#       if (GRID_SIMD)
          for (int lane = 0; lane < n_simd_lanes; lane++)
          {
            outputFile_J << J[index(p,f)].getlane(lane) << "\t";
          }
#       else
          outputFile_J << J[index(p,f)] << "\t";
#       endif
      }

      outputFile_J << endl;
    }

    outputFile_J.close ();


    const string file_name_bc = output_folder + "bc" + tag + ".txt";

    ofstream outputFile_bc (file_name_bc);

    //for (long r = 0; r < nrays_red; r++)
    //{
      long r = 0;
      for (long b = 0; b < nboundary; b++)
      {
        for (long f = 0; f < nfreq_red; f++)
        {
#         if (GRID_SIMD)
            for (int lane = 0; lane < n_simd_lanes; lane++)
            {
              outputFile_bc << boundary_intensity[r][b][f].getlane(lane) << "\t";
            }
#         else
            outputFile_bc << boundary_intensity[r][b][f] << "\t";
#         endif
        }
          outputFile_bc << endl;
      }
    //}

    outputFile_bc.close ();

  }


  return (0);

}
