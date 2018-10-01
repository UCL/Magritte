// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __FREQUENCIES_HPP_INCLUDED__
#define __FREQUENCIES_HPP_INCLUDED__

#include <vector>
using namespace std;

#include "types.hpp"
#include "GridTypes.hpp"
#include "temperature.hpp"
#include "Lines/src/linedata.hpp"


struct FREQUENCIES
{

  const long ncells;   ///< number of cells

  const long nlines;         ///< number of lines
  const long nfreq;          ///< number of frequencies
  const long nfreq_red;      ///< nfreq divided by n_simd_lanes

  vReal2  nu;                ///< [Hz] frequencies (ordered in f) (p,f)
  vReal2 dnu;                ///< [Hz] size of the frequency intervals

  Long4 nr_line;             ///< frequency number corresponing to line (p,l,k,z)

  Double1 line;              ///< [Hz] ordered line center frequencies
  Long1   line_index;        ///< (l,k) index of the corresponding frequency in line


  FREQUENCIES (const long      num_of_cells,
               const LINEDATA &linedata     );           ///< Constructor

  static long count_nlines (const LINEDATA& linedata);   ///< count nr of lines

  static long count_nfreq (const long nlines);           ///< count nr of frequencies

  static long count_nfreq_red (const long nfreq);        ///< count reduced nr of frequencies

  int write (const string tag) const;                                ///< write out data

  int reset (const LINEDATA    &linedata,
             const TEMPERATURE &temperature);            ///< Set frequencies

};


#endif // __FREQUENCIES_HPP_INCLUDED__
