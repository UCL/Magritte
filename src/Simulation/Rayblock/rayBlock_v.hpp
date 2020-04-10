// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __RAYBLOCK_V_HPP_INCLUDED__
#define __RAYBLOCK_V_HPP_INCLUDED__

#include "Model/model.hpp"
#include "Tools/timer.hpp"


typedef double Real;
typedef size_t Size;


struct ProtoRayBlock_v
{
    const Size depth;

    vector<RayData> rays_ar;   ///< data for the antipodal ray
    vector<RayData> rays_rr;   ///< data for the regular ray
    vector<Size>    origins;   ///< origin of the ray

    ProtoRayBlock_v (const RayData &ray_ar, const RayData &ray_rr, const Size origin)
            : depth (ray_ar.size() + ray_rr.size() + 1)
    {
        rays_ar.push_back (ray_ar);
        rays_rr.push_back (ray_rr);
        origins.push_back (origin);
    }


    inline void add (const RayData &ray_ar, const RayData &ray_rr, const Size origin)
    {
        assert (depth == ray_ar.size() + ray_rr.size() + 1);

        rays_ar.push_back (ray_ar);
        rays_rr.push_back (ray_rr);
        origins.push_back (origin);
    }

    inline Size nraypairs () const
    {
        assert (origins.size() == rays_ar.size());
        assert (origins.size() == rays_rr.size());

        return origins.size();
    }

};


///  RayBlock_v: data structure for a pair of rays
//////////////////////////////////////////////////

class RayBlock_v
{

public:

    Real inverse_dtau_max = 1.0;

    Timer timer0 = Timer("total");
    Timer timer1 = Timer("set_frq");
    Timer timer2 = Timer("run_u-d");
    Timer timer3 = Timer("mem_cpy");

    const Size ncells;           ///< total number of cells
    const Size nfreqs;           ///< total number of frequency bins
    const Size nlines;           ///< total number of lines

    const Size nraypairs_max;    ///< maximum number or ray pairs in the ray block
          Size nraypairs;        ///< number or ray pairs in the ray block

    const Size depth_max;        ///< maximum depth of the ray block (lengths of the ray pairs)
    const Size width_max;        ///< maximum width of the ray block (nraypairs_max * nfreqs)
          Size width;            ///< width of the ray block (nraypairs * nfreqs)


    /// Constructor
    RayBlock_v (
          const Size ncells,
          const Size nfreqs,
          const Size nlines,
          const Size nraypairs,
          const Size depth     );

    /// Destructor
    ~RayBlock_v ();


    /// Setup
    void copy_model_data (const Model &model);

    void setup (
        const Model           &model,
        const Size             R,
        const Size             r,
        const ProtoRayBlock_v &prb   );

    void solve ();

    void store (Model &model) const;


    void solve_Feautrier (const Size w);


private:

      Size n_off_diag = 0;   ///< number of off-diagonal rows on one side (default = 0)

      Size RR;               ///< absolute index of the ray direction
      Size rr;               ///< relative index of the ray direction

      Size first;            ///< index of the first used element
      Size last;             ///< index of the last used element

      Size *n1;              ///< number of encountered cells in ray 1
      Size *n2;              ///< number of encountered cells in ray 2

      Size n1_min;           ///< minimum of n1

      Size *origins;         ///< origins of the ray pairs in the ray block
      Real *reverse;         ///< 1.0 if ray_ar is longer than ray_r (-1.0 otherwise)



    /// Pointers to model data
      double *line;
      double *line_emissivity;
      double *line_opacity;
      double *line_width;

      double *frequencies;

      Size *nrs;                  ///< cell number corresponding to point along the ray
      Real *shifts;               ///< Doppler shift scale factors along the ray
      Real *dZs;                  ///< distance increments corresponding to point along the ray

      Real *term1;                ///< effective source for u along the ray
      Real *term2;                ///< effective source for v along the ray

      Real *eta;                  ///< emissivities
      Real *chi;                  ///< opacities

      Real *A;                    ///< A coefficient in Feautrier recursion relation
      Real *a;                    ///< a coefficient in Feautrier recursion relation
      Real *C;                    ///< C coefficient in Feautrier recursion relation
      Real *c;                    ///< c coefficient in Feautrier recursion relation
      Real *F;                    ///< helper variable from Rybicki & Hummer (1991)
      Real *G;                    ///< helper variable from Rybicki & Hummer (1991)

      Real *inverse_A;            ///< helper variable
      Real *inverse_C;            ///< helper variable
      Real *inverse_one_plus_F;   ///< helper variable
      Real *inverse_one_plus_G;   ///< helper variable
      Real * G_over_one_plus_G;   ///< helper variable

      Real *Su;                   ///< effective source for u along the ray
      Real *Sv;                   ///< effective source for v along the ray
      Real *dtau;                 ///< optical depth increment along the ray

      // vReal2 L_upper;   ///< upper-half of L matrix
      Real *L_diag;    ///< diagonal   of L matrix
      // vReal2 L_lower;   ///< lower-half of L matrix

      /// Indices
      inline Size I (const Size i, const Size w) const {return w + i*width;    };
      inline Size D (const Size i, const Size d) const {return d + i*depth_max;};
      inline Size L (const Size i, const Size l) const {return l + i*nlines;   };
      inline Size V (const Size i, const Size f) const {return f + i*nfreqs;   };

    /// Setters
//      CUDA_HOST void setFrequencies (
//          const Double1 &frequencies,
//          const Real     scale,
//          const Size     index,
//          const Size     rp          );

      /// Getter
//      CUDA_DEVICE void get_eta_and_chi (
//          const Size In,
//          const Size Dn,
//          const Real frequency         );

    void get_eta_and_chi (
        const Size  Dn,
        const Real  frequency,
              Real &eta,
              Real &chi          );

//    CUDA_DEVICE void get_eta_and_chi (
//            const Size d,
//            const Size f,
//            const Size rp,
//            const Real frequency         );

      /// Interpolator
//      CUDA_DEVICE Real frequency_interpolate (
//          const Real *Vs,
//          const Size  i,
//          const Size  w                      );

  //    inline double get_L_diag (
  //        const Thermodynamics &thermodynamics,
  //        const double          inverse_mass,
  //        const double          freq_line,
  //        const int             lane           ) const;

  //    inline double get_L_lower (
  //        const Thermodynamics &thermodynamics,
  //        const double          inverse_mass,
  //        const double          freq_line,
  //        const int             lane,
  //        const long            m              ) const;

  //    inline double get_L_upper (
  //        const Thermodynamics &thermodynamics,
  //        const double          inverse_mass,
  //        const double          freq_line,
  //        const int             lane,
  //        const long            m              ) const;

  //    inline void update_Lambda (
  //        const Frequencies    &frequencies,
  //        const Thermodynamics &thermodynamics,
  //        const long            p,
  //        const long            f,
  //        const double          weight_angular,
  //              Lines          &lines          ) const;

};


#endif // __RAYBLOCK_V_HPP_INCLUDED__
