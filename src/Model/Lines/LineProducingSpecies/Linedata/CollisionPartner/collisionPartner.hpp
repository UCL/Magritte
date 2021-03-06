// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __COLLISIONPARTNER_HPP_INCLUDED__
#define __COLLISIONPARTNER_HPP_INCLUDED__


#include "Io/io.hpp"
#include "Tools/types.hpp"


struct CollisionPartner
{

  size_t  num_col_partner;   ///< species number corresponding to collision partner
  string  orth_or_para_H2;   ///< stores whether it is ortho or para (if it is H2)

  size_t ntmp;               ///< number of defined temperatures
  size_t ncol;               ///< number of collisional transitions

  Long1 icol;                ///< level index of collisional transition
  Long1 jcol;                ///< level index of collisional transition

  Double1 tmp;               ///< Collision temperatures for each partner

  Double2 Ce;                ///< Collisional excitation rates for each temperature
  Double2 Cd;                ///< Collisional de-excitation rates for each temperature

  Double1 Ce_intpld;        ///< interpolated Collisional excitation
  Double1 Cd_intpld;        ///< interpolated Collisional de-excitation


  // Io
  void read  (const Io &io, const int l, const int c);
  void write (const Io &io, const int l, const int c) const;

  // Inlined functions
  inline void adjust_abundance_for_ortho_or_para (
      const double  temperature_gas,
            double &abundance                    ) const;

  inline void interpolate_collision_coefficients (
        const double temperature_gas             );


  private:

      static const string prefix;

};


#include "collisionPartner.tpp"


#endif // __COLLISIONPARTNER_HPP_INCLUDED__
