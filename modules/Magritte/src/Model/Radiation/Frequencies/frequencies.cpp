// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <string>

#include "frequencies.hpp"
#include "Tools/constants.hpp"
#include "Tools/Parallel/wrap_Grid.hpp"
#include "Tools/Parallel/wrap_omp.hpp"

#include <iostream>
using namespace std;

const string Frequencies::prefix = "Radiation/Frequencies/";


///  read: read in the data file
///    @param[in] io: io object
///    @paran[in] parameters: model parameters object
/////////////////////////////////////////////////////

int Frequencies ::
    read (
        const Io         &io,
              Parameters &parameters)
{

  ncells = parameters.ncells ();
  nlines = parameters.nlines ();
  nquads = parameters.nquads ();


  // Count line frequencies
  nfreqs = nlines * nquads;

  // Add extra frequency bins around lines to get nicer spectrum
  //nfreqs += nlines * 2 * nbins;

  // Add ncont bins background
  //nfreqs += ncont;

  // Ensure that nfreq is a multiple of n_simd_lanes
  nfreqs_red = reduced (nfreqs);
  nfreqs     = nfreqs_red * n_simd_lanes;


  parameters.set_nfreqs     (nfreqs);
  parameters.set_nfreqs_red (nfreqs_red);


  nu.resize (ncells);

  for (long p = 0; p < ncells; p++)
  {
    nu[p].resize (nfreqs_red);
  }


  // frequencies.nu has to be initialized (for unused entries)

  OMP_PARALLEL_FOR (p, ncells)
  {
    for (long f = 0; f < nfreqs_red; f++)
    {
       nu[p][f] = 0.0;
    }
  }


  return (0);

}




///  write: write out data structure
///    @param[in] io: io object
////////////////////////////////////

int Frequencies ::
    write (
        const Io &io) const
{

  // Print all frequencies (nu)
# if (GRID_SIMD)

    Double2 nu_expanded (ncells, Double1 (nfreqs));


    OMP_PARALLEL_FOR (p, ncells)
    {
      long index = 0;

      for (long f = 0; f < nfreqs_red; f++)
      {
        for (int lane = 0; lane < n_simd_lanes; lane++)
        {
          nu_expanded[p][index] = nu[p][f].getlane (lane);
          index++;
        }
      }
    }

    io.write_array (prefix+"nu", nu_expanded);

# else

    io.write_array (prefix+"nu", nu);

# endif


  return (0);

}
