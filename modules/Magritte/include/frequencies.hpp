// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __FREQUENCIES_HPP_INCLUDED__
#define __FREQUENCIES_HPP_INCLUDED__


#include "io.hpp"
#include "types.hpp"
#include "GridTypes.hpp"
#include "temperature.hpp"


struct Frequencies
{

  long ncells;       ///< number of cells
  long nlspecs;      ///< number of cells

  long nlines;       ///< number of lines
  // const long nbins = 0;    ///< number of extra bins per line
  // const long ncont = 0;    ///< number of background bins

  long nfreqs;       ///< number of frequencies
  long nfreqs_red;   ///< nfreq divided by n_simd_lanes

  Long1 nrad;

  vReal2 nu;              ///< [Hz] frequencies (ordered in f) (p,f)

  Long4 nr_line;          ///< frequency number corresponing to line (p,l,k,z)

  Double1 line;           ///< [Hz] line center frequencies orderd
  Long1   line_index;     ///< index of the corresponding frequency in line


  // Setup and I/O
  int read (
      const Io &io);                    ///< write out data

  int write (
      const Io &io) const;            ///< write out data


  int reset (
      const Temperature &temperature);            ///< Set frequencies

};


#endif // __FREQUENCIES_HPP_INCLUDED__
