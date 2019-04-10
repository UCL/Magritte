// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <iostream>
#include <iomanip>

#include "radiation.hpp"
#include "Tools/constants.hpp"
#include "Tools/Parallel/wrap_omp.hpp"
#include "Tools/Parallel/wrap_mpi.hpp"
#include "Tools/Parallel/wrap_Grid.hpp"
#include "Tools/logger.hpp"


///  read: read radiation field from file
/////////////////////////////////////////

int Radiation ::
    read (
        const Io         &io,
              Parameters &parameters)
{

  write_to_log ("Reading radiation");


  frequencies.read (io, parameters);


  ncells     = parameters.ncells     ();
  nrays      = parameters.nrays      ();
  nfreqs_red = parameters.nfreqs_red ();
  nboundary  = parameters.nboundary  ();


  nrays_red = MPI_length (nrays/2);


  parameters.set_nrays_red (nrays_red);


  // Size and initialize u, v, U and V

  u.resize (nrays_red);
  v.resize (nrays_red);

  U.resize (nrays_red);
  V.resize (nrays_red);

  I_bdy.resize (nrays_red);

  for (long r = 0; r < nrays_red; r++)
  {
    u[r].resize (ncells*nfreqs_red);
    v[r].resize (ncells*nfreqs_red);

    U[r].resize (ncells*nfreqs_red);
    V[r].resize (ncells*nfreqs_red);

    I_bdy[r].resize (ncells);

    for (long p = 0; p < nboundary; p++)
    {
      I_bdy[r][p].resize (nfreqs_red);
    }
  }

  J.resize (ncells*nfreqs_red);
  G.resize (ncells*nfreqs_red);


  return (0);

}




int Radiation ::
    write (
        const Io &io) const
{

  write_to_log ("Writing radiation");


  frequencies.write (io);

  // Print all frequencies (nu)
//# if (GRID_SIMD)
//
//    Double3 u_expanded (ncells, Double1 (ncells*nfreqs));
//
//
//    OMP_PARALLEL_FOR (p, ncells)
//    {
//      long index = 0;
//
//      for (long f = 0; f < nfreqs_red; f++)
//      {
//        for (int lane = 0; lane < n_simd_lanes; lane++)
//        {
//          nu_expanded[p][index] = nu[p][f].getlane (lane);
//          index++;
//        }
//      }
//    }
//
//    io.write_array (prefix+"nu", nu_expanded);
//
//# else
//
//    io.write_array (prefix+"nu", nu);
//
//# endif
//  if (MPI_comm_rank () == 0)
//  {
//    const string file_name_J = output_folder + "J" + tag + ".txt";
//    const string file_name_G = output_folder + "G" + tag + ".txt";
//
//    ofstream outputFile_J (file_name_J);
//    ofstream outputFile_G (file_name_G);
//
//    outputFile_J << scientific << setprecision(16);
//    outputFile_G << scientific << setprecision(16);
//
//
//    for (long p = 0; p < ncells; p++)
//    {
//      for (int f = 0; f < nfreq_red; f++)
//      {
//#       if (GRID_SIMD)
//          for (int lane = 0; lane < n_simd_lanes; lane++)
//          {
//            outputFile_J << J[index(p,f)].getlane(lane) << "\t";
//            outputFile_G << G[index(p,f)].getlane(lane) << "\t";
//          }
//#       else
//          outputFile_J << J[index(p,f)] << "\t";
//          outputFile_G << G[index(p,f)] << "\t";
//#       endif
//      }
//
//      outputFile_J << endl;
//      outputFile_G << endl;
//    }
//
//    outputFile_J.close ();
//    outputFile_G.close ();
//
//
//    const string file_name_bc = output_folder + "bc" + tag + ".txt";
//
//    ofstream outputFile_bc (file_name_bc);
//
//    //for (long r = 0; r < nrays_red; r++)
//    //{
//      long r = 0;
//      for (long b = 0; b < nboundary; b++)
//      {
//        for (long f = 0; f < nfreq_red; f++)
//        {
//#         if (GRID_SIMD)
//            for (int lane = 0; lane < n_simd_lanes; lane++)
//            {
//              outputFile_bc << boundary_intensity[r][b][f].getlane(lane) << "\t";
//            }
//#         else
//            outputFile_bc << boundary_intensity[r][b][f] << "\t";
//#         endif
//        }
//          outputFile_bc << endl;
//      }
//    //}
//
//    outputFile_bc.close ();
//
//  }
//
//
  return (0);

}




int initialize (
    vReal1 &vec)
{

  OMP_PARALLEL_FOR (i, vec.size())
  {
    vec[i] = 0.0;
  }


  return (0);

}




int Radiation ::
    calc_J_and_G (
        const Double2 weights)

#if (MPI_PARALLEL)

{

  initialize (J);
  initialize (G);

  MPI_PARALLEL_FOR (r, nrays/2)
  {
    const long R = r - MPI_start (nrays/2);

    OMP_PARALLEL_FOR (p, ncells)
    {
      for (long f = 0; f < nfreqs_red; f++)
      {
        J[index(p,f)] += 2.0 * weights[p][r] * u[R][index(p,f)];
        G[index(p,f)] += 2.0 * weights[p][r] * v[R][index(p,f)];
      }
    }
  }


  MPI_Datatype MPI_VREAL;
  MPI_Type_contiguous (n_simd_lanes, MPI_DOUBLE, &MPI_VREAL);
  MPI_Type_commit (&MPI_VREAL);

  MPI_Op MPI_VSUM;
  MPI_Op_create ( (MPI_User_function*) mpi_vector_sum, true, &MPI_VSUM);


  int ierr1 = MPI_Allreduce (
                MPI_IN_PLACE,      // pointer to data to be reduced -> here in //place
                J.data(),          // pointer to data to be received
                J.size(),          // size of data to be received
                MPI_VREAL,         // type of reduced data
                MPI_VSUM,          // reduction operation
                MPI_COMM_WORLD);

  assert (ierr1 == 0);


  int ierr2 = MPI_Allreduce (
                MPI_IN_PLACE,      // pointer to data to be reduced -> here in //place
                G.data(),          // pointer to data to be received
                G.size(),          // size of data to be received
                MPI_VREAL,         // type of reduced data
                MPI_VSUM,          // reduction operation
                MPI_COMM_WORLD);

  assert (ierr2 == 0);


  MPI_Type_free (&MPI_VREAL);
  MPI_Op_free   (&MPI_VSUM);


  return (0);

}

#else

{

  initialize (J);
  initialize (G);

  MPI_PARALLEL_FOR (r, nrays/2)
  {
    const long R = r - MPI_start (nrays/2);

    OMP_PARALLEL_FOR (p, ncells)
    {
      for (long f = 0; f < nfreqs_red; f++)
      {
        J[index(p,f)] += 2.0 * weights[p][r] * u[R][index(p,f)];
        G[index(p,f)] += 2.0 * weights[p][r] * v[R][index(p,f)];
      }
    }
  }


  return (0);

}

#endif




int Radiation ::
    calc_U_and_V ()

#if (MPI_PARALLEL)

{

  vReal1 U_local (ncells*nfreqs_red);
  vReal1 V_local (ncells*nfreqs_red);


  MPI_Datatype MPI_VREAL;
  MPI_Type_contiguous (n_simd_lanes, MPI_DOUBLE, &MPI_VREAL);
  MPI_Type_commit (&MPI_VREAL);

  MPI_Op MPI_VSUM;
  MPI_Op_create ( (MPI_User_function*) mpi_vector_sum, true, &MPI_VSUM);


  for (int w = 0; w < MPI_comm_size(); w++)
  {
    const long start = ( w   *(nrays/2)) / MPI_comm_size();
    const long stop  = ((w+1)*(nrays/2)) / MPI_comm_size();

    for (long r1 = start; r1 < stop; r1++)
    {
      const long R1 = r1 - start;

      initialize (U_local);
      initialize (V_local);

      MPI_PARALLEL_FOR (r2, nrays/2)
      {
        const long R2 = r2 - MPI_start (nrays/2);

        OMP_PARALLEL_FOR (p, ncells)
        {
          for (long f = 0; f < nfreqs_red; f++)
      	  {
            U_local[index(p,f)] += u[R2][index(p,f)] * //scattering.phase[r1][r2][f];
            V_local[index(p,f)] += v[R2][index(p,f)] * //scattering.phase[r1][r2][f];
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
  MPI_Op_free   (&MPI_VSUM);


  return (0);

}

#else

{

  vReal1 U_local (ncells*nfreqs_red);
  vReal1 V_local (ncells*nfreqs_red);

  for (long r1 = 0; r1 < nrays/2; r1++)
  {
    initialize (U_local);
    initialize (V_local);

    for (long r2 = 0; r2 < nrays/2; r2++)
    {
      OMP_PARALLEL_FOR (p, ncells)
      {
        for (long f = 0; f < nfreqs_red; f++)
        {
          U_local[index(p,f)] += u[r2][index(p,f)] ;//* scattering.phase[r1][r2][f];
          V_local[index(p,f)] += v[r2][index(p,f)] ;//* scattering.phase[r1][r2][f];
        }
      }
    }

    U[r1] = U_local;
    V[r1] = V_local;
  }


  return (0);

}

#endif
