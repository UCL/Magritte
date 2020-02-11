// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __QUADRATURE_HPP_INCLUDED__
#define __QUADRATURE_HPP_INCLUDED__


#include "Io/io.hpp"
#include "Tools/types.hpp"
#include "Model/parameters.hpp"


struct Quadrature
{

  public:

      Double1 roots;
      Double1 weights;


      // Io
      void read  (const Io &io, const int l, Parameters &parameters);
      void write (const Io &io, const int l                        ) const;


  private:

      size_t nquads;   ///< number frequency quadrature points

      static const string prefix;

};


#endif // __QUADRATURE_HPP_INCLUDED__
