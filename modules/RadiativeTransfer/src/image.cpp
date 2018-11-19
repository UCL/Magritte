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
#include "mpiTools.hpp"


///  Constructor for IMAGE
//////////////////////////

IMAGE ::
IMAGE (const long num_of_cells,
       const long num_of_rays,
       const long num_of_freq_red)
  : ncells    (num_of_cells)
  , nrays     (num_of_rays)
  , nrays_red (MPI_length(nrays/2))
  , nfreq_red (num_of_freq_red)
{

  // Size and initialize Ip_out and Im_out

  I_p.resize (nrays_red);
  I_m.resize (nrays_red);

  for (long r = 0; r < nrays_red; r++)
  {
    I_p[r].resize (ncells);
    I_m[r].resize (ncells);

    for (long p = 0; p < ncells; p++)
    {
      I_p[r][p].resize (nfreq_red);
      I_m[r][p].resize (nfreq_red);
    }
  }


}   // END OF CONSTRUCTOR




///  print: write out the images
///    @param[in] tag: tag for output file
//////////////////////////////////////////

int IMAGE ::
    print (const string tag) const
{

  for (long r = MPI_start(nrays/2); r < MPI_stop(nrays/2); r++)
  {
    const long R = r - MPI_start(nrays/2);

    const string file_name_p = output_folder + "image_p_" + to_string(r) + tag + ".txt";
    const string file_name_m = output_folder + "image_m_" + to_string(r) + tag + ".txt";

    ofstream outputFile_p (file_name_p);
    ofstream outputFile_m (file_name_m);

    outputFile_p << scientific << setprecision(16);
    outputFile_m << scientific << setprecision(16);


    for (long p = 0; p < ncells; p++)
    {
      for (int f = 0; f < nfreq_red; f++)
      {
#       if (GRID_SIMD)
          for (int lane = 0; lane < n_simd_lanes; lane++)
          {
            outputFile_p << I_p[R][p][f].getlane(lane) << "\t";
            outputFile_m << I_m[R][p][f].getlane(lane) << "\t";
          }
#       else
          outputFile_p << I_p[R][p][f] << "\t";
          outputFile_m << I_m[R][p][f] << "\t";
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
