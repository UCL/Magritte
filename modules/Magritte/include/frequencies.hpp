// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __FREQUENCIES_HPP_INCLUDED__
#define __FREQUENCIES_HPP_INCLUDED__


#include "types.hpp"
#include "GridTypes.hpp"
#include "temperature.hpp"
#include "linedata.hpp"


struct Frequencies
{

  const long ncells;       ///< number of cells
  const long nlspecs;      ///< number of cells

  const long nlines;       ///< number of lines
  // const long nbins = 0;    ///< number of extra bins per line
  // const long ncont = 0;    ///< number of background bins

  const long nfreqs;       ///< number of frequencies
  const long nfreqs_red;   ///< nfreq divided by n_simd_lanes

  vReal2 nu;              ///< [Hz] frequencies (ordered in f) (p,f)

  Long4 nr_line;          ///< frequency number corresponing to line (p,l,k,z)

  Double1 line;           ///< [Hz] line center frequencies orderd
  Long1   line_index;     ///< index of the corresponding frequency in line

  // Constructor
  Frequencies (
      const long number_of_cells,
      const long number_of_line_species,
      const long number_of_freqs );           ///< Constructor

  // Setup and I/O
  int read (
      const string input_folder);                    ///< write out data

  int write (
      const string output_folder,
      const string tag           ) const;            ///< write out data

  int setup ();


  static long count_nlines (
      const Linedata &linedata);   ///< count nr of lines

  static long count_nfreq (
      const long nlines,
      const long nbins,
      const long ncont  );          ///< count nr of frequencies


  int reset (
      const Temperature &temperature);            ///< Set frequencies

};


#endif // __FREQUENCIES_HPP_INCLUDED__
