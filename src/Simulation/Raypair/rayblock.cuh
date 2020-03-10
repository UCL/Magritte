// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __RAYBLOCK_CUH_INCLUDED__
#define __RAYBLOCK_CUH_INCLUDED__

#include "Model/model.hpp"
#include "myCudaTools.cuh"
#include "Tools/timer.hpp"

#ifdef __CUDACC__
# define CUDA_HOST_DEVICE __host__ __device__
# define CUDA_HOST        __host__
# define CUDA_DEVICE      __device__
# define CUDA_GLOBAL      __global__
#else
# define CUDA_HOST_DEVICE
# define CUDA_HOST
# define CUDA_DEVICE
#endif


typedef double Real;
typedef size_t Size;


struct ProtoRayBlock
{
    const Size depth;

    vector<RayData> rays_ar;   ///< data for the antipodal ray
    vector<RayData> rays_rr;   ///< data for the regular ray
    vector<Size>    origins;   ///< origin of the ray

    ProtoRayBlock (const RayData &ray_ar, const RayData &ray_rr, const Size origin)
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




struct RayQueue
{
    const Size                    desired_nraypairs;   ///< desired number of ray pairs in a block
    list<ProtoRayBlock>           queue;               ///< list containing the queued proto ray blocks
    list<ProtoRayBlock>::iterator complete_it;         ///< iterator to the complete proto ray block
    bool                          complete = false;    ///< indicates whether there is a complete prb

    RayQueue (const Size nraypairs): desired_nraypairs (nraypairs) {};

    inline void add (
        const RayData &ray_ar,
        const RayData &ray_rr,
        const Size origin,
        const Size depth      )
    {
        for (auto it = queue.begin(); it != queue.end(); it++)
        {
            auto &prb = *it;

            if (prb.depth == depth)
            {
                prb.add (ray_ar, ray_rr, origin);

                if (prb.nraypairs() == desired_nraypairs)
                {
                    complete_it = it;
                    complete    = true;
                }

                return;
            }
        }

        queue.push_back (ProtoRayBlock (ray_ar, ray_rr, origin));

        if (queue.back().nraypairs() == desired_nraypairs)
        {
            auto last = queue.end();
            last--;

            complete_it = last;
            complete    = true;
        }

        return;
    }


    inline ProtoRayBlock get_complete_block ()
    {
        const ProtoRayBlock complete_block = *complete_it;

        queue.erase (complete_it);

        complete = false;

        return complete_block;
    }
};




///  Raypair: data structure for a pair of rays
///////////////////////////////////////////////

class RayBlock : public Managed
{

  public:

      Timer timer0 = Timer("total");
      Timer timer1 = Timer("set_frq");
      Timer timer2 = Timer("run_u-d");
      Timer timer3 = Timer("mem_cpy");

      Size gpuBlockSize = 32;

      const Size ncells;           ///< total number of cells
      const Size nfreqs;           ///< total number of frequency bins
      const Size nlines;           ///< total number of lines

      const Size nraypairs_max;    ///< maximum number or ray pairs in the ray block
            Size nraypairs;        ///< number or ray pairs in the ray block

      const Size depth_max;        ///< maximum depth of the ray block (lengths of the ray pairs)
      const Size width_max;        ///< maximum width of the ray block (nraypairs_max * nfreqs)
            Size width;            ///< width of the ray block (nraypairs * nfreqs)


    /// Constructor
      CUDA_HOST  RayBlock (
          const Size ncells,
          const Size nfreqs,
          const Size nlines,
          const Size nraypairs,
          const Size depth     );

    /// Destructor
      CUDA_HOST ~RayBlock ();

      /// Setup
      CUDA_HOST void copy_model_data (const Model &model);

      CUDA_HOST void setup (
          const Model         &model,
          const Size           R,
          const Size           r,
          const ProtoRayBlock &prb   );

      CUDA_HOST void solve ();

      CUDA_HOST void store (Model &model) const;


      CUDA_DEVICE void solve_Feautrier (const Size w);


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

      Real *I_bdy_0_presc;        ///< boundary intensity at the first ray element
      Real *I_bdy_n_presc;        ///< boundary intensity at the final ray element

      Size *nrs;                  ///< cell number corresponding to point along the ray
      Real *shifts;               ///< Doppler shift scale factors along the ray
      Real *dZs;                  ///< distance increments corresponding to point along the ray

//      Real *freqs;
//      Real *freqs_scaled;
//      Size *freqs_lower;
//      Size *freqs_upper;

      Real *term1;                ///< effective source for u along the ray
      Real *term2;                ///< effective source for v along the ray

      Real *eta;                  ///< emissivities
      Real *chi;                  ///< opacities

      Real *A;                    ///< A coefficient in Feautrier recursion relation
      Real *C;                    ///< C coefficient in Feautrier recursion relation
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
      CUDA_HOST_DEVICE inline Size I (const Size i, const Size w) const {return w + i*width;    };
      CUDA_HOST_DEVICE inline Size D (const Size i, const Size d) const {return d + i*depth_max;};
      CUDA_HOST_DEVICE inline Size L (const Size i, const Size l) const {return l + i*nlines;   };
      CUDA_HOST_DEVICE inline Size V (const Size i, const Size f) const {return f + i*nfreqs;   };

    /// Setters
//      CUDA_HOST void setFrequencies (
//          const Double1 &frequencies,
//          const Real     scale,
//          const Size     index,
//          const Size     rp          );

      /// Getter
      CUDA_DEVICE void get_eta_and_chi (
          const Size In,
          const Size Dn,
          const Real frequency         );

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


#endif // __RAYBLOCK_CUH_INCLUDED__
