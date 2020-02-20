// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __RAYPAIR_HPP_INCLUDED__
#define __RAYPAIR_HPP_INCLUDED__


#include "Tools/Parallel/wrap_Grid.hpp"
#include "Model/model.hpp"


///  Raypair: data structure for a pair of rays
///////////////////////////////////////////////

struct RayPair
{

    Double1 dtaus;
    Double1 dZs;
    Double1 chis;
    Long2   pre;
    Long2   pos;


  public:

      long n_ar;       ///< number of points on the antipodal ray side
      long n_r;        ///< number of points on the ray side

      long first;      ///< index of the first used element
      long last;       ///< index of the last used element

      long ndep;       ///< total number of points on the ray

      vReal I_bdy_0;   ///< Incoming intensity on the front of the ray (boundary condition)
      vReal I_bdy_n;   ///< Incoming intensity on the back of the ray (boundary condition)

      long lnotch_at_origin;

      vReal1 chi;      ///< opacities
      Long1  nrs;      ///< corresponding cell indices
      vReal1 frs;      ///< frequencies

      long n_off_diag = 0;   ///< number of off-diagonal rows on one side (default = 0)


      //RayPair (
      //    const long length,
      //    const long n_off_diag);

      int resize (
          const long length,
          const long n_off_diag);


      // inline functions
      inline void initialize (
          const long n_ar,
          const long n_r     );

      inline void solve_using_scattering ();
      inline void solve                  ();
      inline void solve_for_image        ();

      //inline void solve_ndep_is_1 ();

      // setters
      inline void set_term1_and_term2 (
          const vReal &eta,
          const vReal &chi,
          const vReal &U_scaled,
          const vReal &V_scaled,
          const long   n              );

      inline void set_term1_and_term2 (
          const vReal &eta,
          const vReal &chi,
          const long   n              );

      inline void set_dtau (
          const vReal  &chi,
          const vReal  &chi_prev,
          const double  dZ,
          const long    n        );

      // getters
      inline vReal get_u_at_origin () const;
      inline vReal get_v_at_origin () const;

      inline vReal get_Ip_at_origin () const;
      inline vReal get_Im_at_origin () const;

      inline vReal get_Ip_at_end   () const;
      inline vReal get_Im_at_front () const;

      inline vReal get_Ip ();
      inline vReal get_Im ();



      inline double get_L_diag (
          const Thermodynamics &thermodynamics,
          const double          inverse_mass,
          const double          freq_line,
          const int             lane           ) const;

      inline double get_L_lower (
          const Thermodynamics &thermodynamics,
          const double          inverse_mass,
          const double          freq_line,
          const int             lane,
          const long            m              ) const;

      inline double get_L_upper (
          const Thermodynamics &thermodynamics,
          const double          inverse_mass,
          const double          freq_line,
          const int             lane,
          const long            m              ) const;

      inline void update_Lambda (
          const Frequencies    &frequencies,
          const Thermodynamics &thermodynamics,
          const long            p,
          const long            f,
          const double          weight_angular,
                Lines          &lines          ) const;

      // The following variables "should" be declared private,
      // but are here for testing purposes...

      vReal1 A;         ///< A coefficient in Feautrier recursion relation
      vReal1 C;         ///< C coefficient in Feautrier recursion relation
      vReal1 F;         ///< helper variable from Rybicki & Hummer (1991)
      vReal1 G;         ///< helper variable from Rybicki & Hummer (1991)


      vReal1 Su;        ///< effective source for u along the ray
      vReal1 Sv;        ///< effective source for v along the ray
      vReal1 dtau;      ///< optical depth increment along the ray

      vReal2 L_upper;   ///< upper-half of L matrix
      vReal1 L_diag;    ///< diagonal   of L matrix
      vReal2 L_lower;   ///< lower-half of L matrix


  private:

      vReal1 term1;   ///< effective source for u along the ray
      vReal1 term2;   ///< effective source for v along the ray

      vReal inverse_dtau0;         ///< helper variable
      vReal inverse_dtaud;         ///< helper variable

      vReal B0;                    ///< helper variable
      vReal inverse_B0;            ///< helper variable
      vReal B0_min_C0;             ///< helper variable

      vReal Bd;                    ///< helper variable
      vReal Bd_min_Ad;             ///< helper variable

      vReal denominator;           ///< helper variable

      vReal1 inverse_one_plus_F;   ///< helper variable
      vReal1 inverse_one_plus_G;   ///< helper variable
      vReal1  G_over_one_plus_G;   ///< helper variable
      vReal1 inverse_A;            ///< helper variable
      vReal1 inverse_C;            ///< helper variable


};


#include "raypair.tpp"
#include "raypair_solver.tpp"
#include "raypair_lambda.tpp"



class rayPair
{

  public:

    const std::size_t ncells;
    const std::size_t nfreqs;

    rayPair(const std::size_t ncs, const std::size_t nfs);
    ~rayPair();

    virtual void setup() = 0;
    virtual void solve() = 0;
    virtual void store() = 0;



};

#endif // __RAYPAIR_HPP_INCLUDED__
