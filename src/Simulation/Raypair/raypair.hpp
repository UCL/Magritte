// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __RAYPAIR_HPP_INCLUDED__
#define __RAYPAIR_HPP_INCLUDED__


#include "Tools/Parallel/wrap_Grid.hpp"


///  Raypair: data structure for a pair of rays
///////////////////////////////////////////////

struct RayPair
{

  public:

      long n_ar;
      long n_r;

      long ndep;

      vReal I_bdy_0;
      vReal I_bdy_n;

      long lnotch_at_origin;
      vReal   eta_at_origin;
      vReal   chi_at_origin;


      // inline functions
      inline void initialize (
          const long n_ar,
          const long n_r  );

      inline void solve ();

      //inline void solve_ndep_is_1 ();

      // setters
      inline void set_term1_and_term2 (
              const vReal &eta,
              const vReal &chi,
              const vReal &U_scaled,
              const vReal &V_scaled,
              const long   n          );

      inline void set_dtau (
              const vReal  &chi,
              const vReal  &chi_prev,
              const double  dZ,
              const long    n        );


      // getters
      inline vReal get_u_at_origin () const;
      inline vReal get_v_at_origin () const;

      inline vReal get_u_local_at_origin () const;

      inline vReal get_I_p () const;
      inline vReal get_I_m () const;


  private:

      vReal1 A;       // A coefficient in Feautrier recursion relation
      vReal1 C;       // C coefficient in Feautrier recursion relation
      vReal1 F;       // helper variable from Rybicki & Hummer (1991)
      vReal1 G;       // helper variable from Rybicki & Hummer (1991)

      vReal1 term1;   // effective source for u along the ray
      vReal1 term2;   // effective source for v along the ray

      vReal1 Su;     // effective source for u along the ray
      vReal1 Sv;     // effective source for v along the ray
      vReal1 dtau;   // optical depth increment along the ray

      vReal1 L;


};


#include "raypair.tpp"


#endif // __RAYPAIR_HPP_INCLUDED__
