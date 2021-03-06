// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __TURBULENCE_HPP_INCLUDED__
#define __TURBULENCE_HPP_INCLUDED__


#include "Io/io.hpp"
#include "Tools/types.hpp"
#include "Model/parameters.hpp"


struct Turbulence
{

  public:
      
      Double1 vturb2;     ///< [.] microturbulence over c all squared


      // Io
      void read  (const Io &io, Parameters &parameters);
      void write (const Io &io                        ) const;


  private:

      size_t ncells;

      static const string prefix;


};


#endif // __TURBULENCE_HPP_INCLUDED__
