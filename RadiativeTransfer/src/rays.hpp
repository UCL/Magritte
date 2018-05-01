// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __RAYS_HPP_INCLUDED__
#define __RAYS_HPP_INCLUDED__

#include "declarations.hpp"
#include "HEALPix/chealpix.h"


///  RAYS: data struct containing directional discretization info
/////////////////////////////////////////////////////////////////

template <int dimension, long Nrays>
struct RAYS
{

  double x[Nrays];         ///< x component of direction vector
  double y[Nrays];         ///< y component of direction vector
  double z[Nrays];         ///< z component of direction vector

  long antipod[Nrays];     ///< ray number of antipodal ray

  long mirror_xz[Nrays];   ///< ray mirrored along xz plane

  RAYS ();                 ///< Constructor

};


#include "rays.tpp"   // Implementation of template functions


#endif // __RAYS_HPP_INCLUDED__
