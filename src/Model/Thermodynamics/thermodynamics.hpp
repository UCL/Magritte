// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __THERMODYNAMICS_HPP_INCLUDED__
#define __THERMODYNAMICS_HPP_INCLUDED__


#include "Io/io.hpp"
#include "Tools/types.hpp"
#include "Tools/Parallel/wrap_Grid.hpp"
#include "Model/parameters.hpp"
#include "Model/Thermodynamics/Temperature/temperature.hpp"
#include "Model/Thermodynamics/Turbulence/turbulence.hpp"


struct Thermodynamics
{

  public:

      Temperature temperature;

      Turbulence  turbulence;


      // Io
      int read (
          const Io         &io,
                Parameters &parameters);

      int write (
          const Io &io) const;

      // Functions
      inline vReal profile (
          const double width,
          const vReal  freq_diff  ) const;

      inline double profile_width (
          const long   p,
          const double freq_line  ) const;

      inline double profile_width (
          const long p            ) const;


};


#include "thermodynamics.tpp"


#endif // __THERMODYNAMICS_HPP_INCLUDED__
