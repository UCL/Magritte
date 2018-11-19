// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __LINES_HPP_INCLUDED__
#define __LINES_HPP_INCLUDED__


#include <vector>
#include <fstream>
using namespace std;
#include <Eigen/Core>
using namespace Eigen;

#include "types.hpp"
#include "GridTypes.hpp"
#include "frequencies.hpp"
#include "temperature.hpp"
#include "Lines/src/linedata.hpp"


///  LINES: bridge between Levels and RadiativeTransfer calculations
////////////////////////////////////////////////////////////////////

struct LINES
{

  const long ncells;          ///< number of cells

  const long nlspec;          ///< number of species producing lines

  const Int1 nrad;
  const Int1 nrad_cum;
  const int  nrad_tot;

  Double1 emissivity;   ///< line emissivity (p,l,k)
  Double1 opacity;      ///< line opacity (p,l,k)


  static Int1 get_nrad_cum (const Int1 nrad);
  static int  get_nrad_tot (const Int1 nrad);


  LINES (const long      num_of_cells,
         const LINEDATA &linedata     );   ///< Constructor


  int print (const string tag) const;

  inline long index (const long p,
                     const int  l,
                     const int  k ) const;

  inline long index (const long p,
                     const long line_index) const;

  inline int add_emissivity_and_opacity (const FREQUENCIES &frequencies,
                                         const TEMPERATURE &temperature,
                                         const vReal       &freq_scaled,
                                               long        &lnotch,
                                         const long         p,
                                               vReal       &eta,
                                               vReal       &chi          ) const;

  int mpi_allgatherv ();

};




#include "profile.hpp"




///  index:
///////////

inline long LINES ::
            index (const long p,
                   const int  l,
                   const int  k) const
{
  return k + nrad_cum[l] + p*nrad_tot;
}



///  index:
///////////

inline long LINES ::
            index                    (
                const long p,
                const long line_index) const
{
  return line_index + p*nrad_tot;
}




///  add_emissivity_and_opacity
///////////////////////////////

inline int LINES ::
           add_emissivity_and_opacity         (
               const FREQUENCIES &frequencies,
               const TEMPERATURE &temperature,
               const vReal       &freq_scaled,
                     long        &lnotch,
               const long         p,
                     vReal       &eta,
                     vReal       &chi          ) const
{
  // TEMPORARY !!!

  lnotch = 0;

  ////////////////


  // Move notch just before first line to include

  vReal freq_diff = freq_scaled - (vReal) frequencies.line[lnotch];
  double    width = profile_width (temperature.gas[p], frequencies.line[lnotch]);


# if (GRID_SIMD)
    while ( (freq_diff.getlane(0) > UPPER*width) && (lnotch < nrad_tot-1) )
# else
    while ( (freq_diff            > UPPER*width) && (lnotch < nrad_tot-1) )
# endif
  {
    lnotch++;

    freq_diff = freq_scaled - (vReal) frequencies.line[lnotch];
        width = profile_width (temperature.gas[p], frequencies.line[lnotch]);
        
       // cout << "LOWER*width = " << LOWER*width << endl;
  }

  // Include lines unt

  long lindex = lnotch;

# if (GRID_SIMD)
    while ( (freq_diff.getlane(n_simd_lanes-1) >= LOWER*width) && (lindex < nrad_tot) )
# else
    while ( (freq_diff                         >= LOWER*width) && (lindex < nrad_tot) )
# endif
  {
    const vReal line_profile = profile (width, freq_diff);
    const long           ind = index   (p, frequencies.line_index[lindex]);

    eta += emissivity[ind] * line_profile;
    chi +=    opacity[ind] * line_profile;

  //  cout  << "l-n = " << lindex - lnotch << "   freq_diff " << freq_diff << "   LW " << LOWER*width << scientific << endl;

    //cout << "FREQ DIFF = " << freq_diff << "    freq_scaled = " << freq_scaled << "   freq_line = " << frequencies.line[lindex]  << endl;
    //cout << "line profile = " << line_profile << endl;
    //
    //cout << eta << " " << chi << endl;

    lindex++;

    freq_diff = freq_scaled - (vReal) frequencies.line[lindex];
        width = profile_width (temperature.gas[p], frequencies.line[lindex]);
  }

  int lmn = lindex - lnotch;

  //if (lmn < 1)
  {
 //   cout  << "l-n = " << lindex - lnotch << "   freq_diff " << freq_diff << "   LW " << LOWER*width << scientific << endl;
  }

  return (0);

}


#endif // __LINES_HPP_INCLUDED__
