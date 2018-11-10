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

#include "image.hpp"
#include "folders.hpp"
#include "constants.hpp"
#include "GridTypes.hpp"
#include "mpiTypes.hpp"


///  Constructor for IMAGE
//////////////////////////

IMAGE ::
IMAGE (const long num_of_cells,
       const long num_of_rays,
       const long num_of_freq_red)
  : ncells        (num_of_cells)
  , nrays         (num_of_rays)
  , nrays_red     (get_nrays_red (nrays))
  , nfreq_red     (num_of_freq_red)
{

  // Size and initialize Ip_out and Im_out

  Ip_out.resize (nrays_red);
  Im_out.resize (nrays_red);

  for (long r = 0; r < nrays_red; r++)
  {
    Ip_out[r].resize (ncells*nfreq_red);
    Im_out[r].resize (ncells*nfreq_red);
  }


}   // END OF CONSTRUCTOR




///  get_nrays_red: get reduced number of rays
///    @param[in] nrays: total number or rays
//////////////////////////////////////////////

long IMAGE ::
     get_nrays_red (const long nrays)
{

  int world_size;
  MPI_Comm_size (MPI_COMM_WORLD, &world_size);

  int world_rank;
  MPI_Comm_rank (MPI_COMM_WORLD, &world_rank);

  const long START_raypair = ( world_rank   *nrays/2)/world_size;
  const long STOP_raypair  = ((world_rank+1)*nrays/2)/world_size;


  return STOP_raypair - START_raypair;
  
}




int IMAGE ::
    print (const string tag) const
{

  for (long R = 0; R < nrays_red; R++)
  {
    const long START_raypair = ( world_rank*nrays/2)/world_size;
    const long             r = R + START_raypair;

    const string file_name_p = output_folder + "image_p_" + to_string(r) + tag + ".txt";
    const string file_name_m = output_folder + "image_m_" + to_string(r) + tag + ".txt";

    ofstream outputFile_p (file_name_p);
    ofstream outputFile_m (file_name_m);

    for (long p = 0; p < ncells; p++)
    {
      for (int f = 0; f < nfreq_red; f++)
      {
#       if (GRID_SIMD)
          for (int lane = 0; lane < n_simd_lanes; lane++)
          {
            outputFile_p << scientific << setprecision(16);
            outputFile_m << scientific << setprecision(16);

            outputFile_p << Ip_out[R][index(p,f)].getlane(lane) << "\t";
            outputFile_m << Im_out[R][index(p,f)].getlane(lane) << "\t";
          }
#       else
          outputFile_p << scientific << setprecision(16);
          outputFile_m << scientific << setprecision(16);

          outputFile_p << Ip_out[R][index(p,f)] << "\t";
          outputFile_m << Im_out[R][index(p,f)] << "\t";
#       endif
      }

      outputFile_p << endl;
      outputFile_m << endl;
    }

    outputFile_p.close ();
    outputFile_m.close ();

  }


  return (0);

}
