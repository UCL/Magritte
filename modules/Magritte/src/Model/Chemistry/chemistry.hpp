// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __CHEMISTRY_HPP_INCLUDED__
#define __CHEMISTRY_HPP_INCLUDED__


#include "Io/io.hpp"
#include "Model/parameters.hpp"
#include "Model/Chemistry/Species/species.hpp"


///  Chemistry: data structure for Chemistry
////////////////////////////////////////////

struct Chemistry
{

  Species  species;


  // Io
  int read (
      const Io         &io,
            Parameters &parameters);

  int write (
      const Io &io) const;


};


#endif // __CHEMISTRY_HPP_INCLUDED__
