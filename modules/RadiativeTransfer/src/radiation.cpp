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
#include "ompTools.hpp"
#include "lines.hpp"
#include "frequencies.hpp"
#include "scattering.hpp"
#include "profile.hpp"


///  Constructor for RADIATION
//////////////////////////////

RADIATION ::
RADIATION (const long num_of_cells,
           const long num_of_rays,
           const long num_of_freq_red,
           const long num_of_bdycells )
  : ncells        (num_of_cells)
  , nrays         (num_of_rays)
  , nrays_red     (MPI_length (nrays/2))
  , nfreq_red     (num_of_freq_red)
  , nboundary     (num_of_bdycells)
{

  // Size and initialize u, v, U and V

  u.resize (nrays_red);
  v.resize (nrays_red);
  Lambda.resize (nrays_red);

  U.resize (nrays_red);
  V.resize (nrays_red);

  boundary_intensity.resize (nrays_red);

  for (long r = 0; r < nrays_red; r++)
  {
    u[r].resize (ncells*nfreq_red);
    v[r].resize (ncells*nfreq_red);
    Lambda[r].resize (ncells*nfreq_red);

    U[r].resize (ncells*nfreq_red);
    V[r].resize (ncells*nfreq_red);

    boundary_intensity[r].resize (ncells);

    for (long p = 0; p < nboundary; p++)
    {
      boundary_intensity[r][p].resize (nfreq_red);
    }
  }

  J.resize (ncells*nfreq_red);
  G.resize (ncells*nfreq_red);
  L.resize (ncells*nfreq_red);

  cell2boundary_nr.resize (ncells);


}   // END OF CONSTRUCTOR




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
    calc_boundary_intensities             (
        const Long1       &Boundary2cell_nr,
        const Long1       &Cell2boundary_nr,
        const FREQUENCIES &frequencies    )
{

  cell2boundary_nr = Cell2boundary_nr;

  for (long r = 0; r < nrays_red; r++)
  {

#   pragma omp parallel                                           \
    shared (r, Boundary2cell_nr, Cell2boundary_nr, frequencies)   \
    default (none)
    {
      for (long b = OMP_start (nboundary); b < OMP_stop (nboundary); b++)
      {
        const long p = Boundary2cell_nr[b];

        for (long f = 0; f < nfreq_red; f++)
        {
          boundary_intensity[r][b][f] = planck (T_CMB, frequencies.nu[p][f]);
        }
      }
    }
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
    for (long i = OMP_start (vec.size()); i < OMP_stop (vec.size()); i++)
    {
      vec[i] = 0.0;
    }
  }


  return (0);

}



int RADIATION ::
    calc_J (void)
{
  const double two_over_nrays = 2.0/nrays;
  const double one_over_nrays = 1.0/nrays;

  initialize (J);
  initialize (G);
  initialize (L);


  for (long r = MPI_start (nrays/2); r < MPI_stop (nrays/2); r++)
  {
    const long R = r - MPI_start (nrays/2);

#   pragma omp parallel   \
    default (none)
    {

    for (long p = OMP_start (ncells); p < OMP_stop (ncells); p++)
    {
      for (long f = 0; f < nfreq_red; f++)
      {
        J[index(p,f)] += two_over_nrays *      u[R][index(p,f)];
        G[index(p,f)] += two_over_nrays *      v[R][index(p,f)];
        L[index(p,f)] += one_over_nrays * Lambda[R][index(p,f)];
      }
    }
    } // end of pragma omp parallel

  } // end of r loop over rays


  MPI_Datatype MPI_VREAL;
  MPI_Type_contiguous (n_simd_lanes, MPI_DOUBLE, &MPI_VREAL);
  MPI_Type_commit (&MPI_VREAL);

  MPI_Op MPI_VSUM;
  MPI_Op_create ( (MPI_User_function*) mpi_vector_sum, true, &MPI_VSUM);


  int ierr1 = MPI_Allreduce (
                MPI_IN_PLACE,      // pointer to data to be reduced -> here in place
                J.data(),          // pointer to data to be received
                J.size(),          // size of data to be received
                MPI_VREAL,         // type of reduced data
                MPI_VSUM,          // reduction operation
                MPI_COMM_WORLD);

  assert (ierr1 == 0);


  int ierr2 = MPI_Allreduce (
                MPI_IN_PLACE,      // pointer to data to be reduced -> here in place
                G.data(),          // pointer to data to be received
                G.size(),          // size of data to be received
                MPI_VREAL,         // type of reduced data
                MPI_VSUM,          // reduction operation
                MPI_COMM_WORLD);

  assert (ierr2 == 0);


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

  if (MPI_comm_rank () == 0)
  {
    const string file_name_J = output_folder + "J" + tag + ".txt";
    const string file_name_G = output_folder + "G" + tag + ".txt";

    ofstream outputFile_J (file_name_J);
    ofstream outputFile_G (file_name_G);

    outputFile_J << scientific << setprecision(16);
    outputFile_G << scientific << setprecision(16);


    for (long p = 0; p < ncells; p++)
    {
      for (int f = 0; f < nfreq_red; f++)
      {
#       if (GRID_SIMD)
          for (int lane = 0; lane < n_simd_lanes; lane++)
          {
            outputFile_J << J[index(p,f)].getlane(lane) << "\t";
            outputFile_G << G[index(p,f)].getlane(lane) << "\t";
          }
#       else
          outputFile_J << J[index(p,f)] << "\t";
          outputFile_G << G[index(p,f)] << "\t";
#       endif
      }

      outputFile_J << endl;
      outputFile_G << endl;
    }

    outputFile_J.close ();
    outputFile_G.close ();


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
