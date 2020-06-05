// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __TEMPERATURE_HPP_INCLUDED__
#define __TEMPERATURE_HPP_INCLUDED__


#include "Io/io.hpp"
#include "Tools/types.hpp"
#include "Model/parameters.hpp"


struct Temperature
{
  public:

      Double1 gas;        ///< [K] gas temperature
//      Double1 dust;       ///< [K] dust temparature
//      Double1 gas_prev;   ///< [K] gas temperature in previous iteration


      // Io
      void read  (const Io &io, Parameters &parameters);
      void write (const Io &io                        ) const;


  private:

      size_t ncells;

      static const string prefix;

};


#endif // __TEMPERATURE_HPP_INCLUDED__
