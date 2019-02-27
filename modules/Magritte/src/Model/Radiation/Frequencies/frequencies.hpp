// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __FREQUENCIES_HPP_INCLUDED__
#define __FREQUENCIES_HPP_INCLUDED__


#include "Io/io.hpp"
#include "Tools/types.hpp"
#include "Tools/Parallel/wrap_Grid.hpp"
#include "Model/Thermodynamics/Temperature/temperature.hpp"


struct Frequencies
{

  public:

      vReal2 nu;              ///< [Hz] frequencies (ordered in f) (p,f)


      // Io
      int read (
          const Io         &io,
                Parameters &parameters);                    ///< write out data

      int write (
          const Io &io) const;            ///< write out data


      //int reset (
      //    const Temperature &temperature);            ///< Set frequencies


  private:

      long ncells;       ///< number of cells
      long nlines;       ///< number of lines
      long nquads;       ///< number frequency quadrature points
      // const long nbins = 0;    ///< number of extra bins per line
      // const long ncont = 0;    ///< number of background bins

      long nfreqs;       ///< number of frequencies
      long nfreqs_red;   ///< nfreq divided by n_simd_lanes

      static const string prefix;


};


#endif // __FREQUENCIES_HPP_INCLUDED__
