// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __COLLISIONPARTNER_HPP_INCLUDED__
#define __COLLISIONPARTNER_HPP_INCLUDED__


#include "types.hpp"
#include "io.hpp"
#include "species.hpp"


struct CollisionPartner
{

  long    num_col_partner;   ///< species number corresponding to collision partner
  string  orth_or_para_H2;   ///< stores whether it is ortho or para (if it is H2)

  long ntmp;                 ///< number of defined temperatures
  long ncol;                 ///< number of collisional transitions

  Long1 icol;                ///< level index of collisional transition
  Long1 jcol;                ///< level index of collisional transition

  Double1 tmp;               ///< Collision temperatures for each partner

  Double2 Ce;                ///< Collisional excitation rates for each temperature
  Double2 Cd;                ///< Collisional de-excitation rates for each temperature


  // Io
  int read (
      const Io &io,
      const int l,
      const int c  );

  int write (
      const Io &io,
      const int l,
      const int c  ) const;

};


#endif // __COLLISIONPARTNER_HPP_INCLUDED__
