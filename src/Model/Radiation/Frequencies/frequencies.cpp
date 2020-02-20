// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <string>

#include "frequencies.hpp"
#include "Tools/constants.hpp"
#include "Tools/Parallel/wrap_Grid.hpp"
#include "Tools/Parallel/wrap_omp.hpp"
#include "Tools/logger.hpp"


const string Frequencies::prefix = "Radiation/Frequencies/";


///  Reader for the Frequencies data
///    @param[in] io         : io object to read with
///    @param[in] parameters : model parameters object
//////////////////////////////////////////////////////

int Frequencies :: read (const Io &io, Parameters &parameters)
{
  cout << "Reading frequencies..." << endl;

  ncells = parameters.ncells ();
  nlines = parameters.nlines ();
  nquads = parameters.nquads ();

  cout << "ncells = " << ncells << endl;
  cout << "nlines = " << nlines << endl;
  cout << "nquads = " << nquads << endl;


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


  appears_in_line_integral.resize (nfreqs);
  corresponding_l_for_spec.resize (nfreqs);
  corresponding_k_for_tran.resize (nfreqs);
  corresponding_z_for_line.resize (nfreqs);


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




///  Writer for the Frequencies data
///    @param[in] io : io object to write with
/////////////////////.////////////////////////

int Frequencies ::
    write (
        const Io &io) const
{

  cout << "Writing frequencies" << endl;


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
