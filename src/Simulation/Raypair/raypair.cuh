// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __RAYPAIR_CUH_INCLUDED__
#define __RAYPAIR_CUH_INCLUDED__

#include "Model/model.hpp"
#include "myCudaTools.cuh"

#ifdef __CUDACC__
# define CUDA_HOST_DEVICE __host__ __device__
# define CUDA_HOST        __host__
# define CUDA_DEVICE      __device__
#else
# define CUDA_HOST_DEVICE
# define CUDA_HOST
# define CUDA_DEVICE
#endif


///  Raypair: data structure for a pair of rays
///////////////////////////////////////////////

class gpuRayPair : public Managed
{

  public:

      const long maxLength;
      const long ncells;
      const long nfreqs;
      const long nlines;

      /// Constructor
      CUDA_HOST  gpuRayPair (
          const long mrl,
          const long ncs,
          const long nfs,
          const long nls    );

      /// Destructor
      CUDA_HOST ~gpuRayPair (void);

      /// Setup
      CUDA_HOST int copy_model_data (
          const Model &model        );

      CUDA_HOST void setup (
          const Model   &model,
          const RayData &raydata1,
          const RayData &raydata2,
          const long     R,
          const long     o          );

      /// Solver
      CUDA_HOST void solve (void);
      /// Solver
      CUDA_DEVICE void solve_Feautrier (
          const long f                 );

      /// Extractor
      CUDA_HOST void extract_radiation_field (
                Model &model,
          const long   R,
          const long   r,
          const long   o                     );


  private:

      double reverse;        ///< 1.0 if ray_ar is longer than ray_r (-1.0 otherwise)

      long n_off_diag = 0;   ///< number of off-diagonal rows on one side (default = 0)

      long   n_ar;           ///< number of points on the antipodal ray side
      long   n_r;            ///< number of points on the ray side

      long   ndep;           ///< total number of points on the ray

      long   first;          ///< index of the first used element
      long   last;           ///< index of the last used element


      /// Pointers to model data
      double *line;
      double *line_emissivity;
      double *line_opacity;
      double *width;

      double3 *position;
      double3 *velocity;

      long    *neighbors;
      long    *n_neighbors;

      double *freqs;
      double *freqs_scaled;
      long   *freqs_lower;
      long   *freqs_upper;

      double *I_bdy_0_presc;
      double *I_bdy_n_presc;

      double *eta;      ///< opacities
      double *chi;      ///< opacities

      long   *nrs;      ///< corresponding cell indices
      double *dZs;      ///< distance increments along the ray

      double *A;         ///< A coefficient in Feautrier recursion relation
      double *C;         ///< C coefficient in Feautrier recursion relation
      double *F;         ///< helper variable from Rybicki & Hummer (1991)
      double *G;         ///< helper variable from Rybicki & Hummer (1991)

      double *Su;        ///< effective source for u along the ray
      double *Sv;        ///< effective source for v along the ray
      double *dtau;      ///< optical depth increment along the ray

          // vReal2 L_upper;   ///< upper-half of L matrix
      double *L_diag;    ///< diagonal   of L matrix
          // vReal2 L_lower;   ///< lower-half of L matrix

      double *term1;   ///< effective source for u along the ray
      double *term2;   ///< effective source for v along the ray

      double *inverse_one_plus_F;   ///< helper variable
      double *inverse_one_plus_G;   ///< helper variable
      double * G_over_one_plus_G;   ///< helper variable
      double *inverse_A;            ///< helper variable
      double *inverse_C;            ///< helper variable

      /// Indices
      CUDA_HOST_DEVICE inline long I (const long i, const long f) const {return f + i*nfreqs;};
      CUDA_HOST_DEVICE inline long L (const long i, const long l) const {return l + i*nlines;};

      /// Setters
      CUDA_HOST void setFrequencies (
          const Double1 &frequencies,
          const double   scale,
          const long     index       );

      /// Getter
      CUDA_DEVICE void get_eta_and_chi (
          const long d,
          const long f                 );
      /// Interpolator
      CUDA_DEVICE double frequency_interpolate (
          const double *Vs,
          const long    f                      );


      CUDA_DEVICE void trace_ray (void);

      CUDA_DEVICE double get_doppler_shift (
          const long     origin,
          const long     current,
          const double3 &ray               ) const;

      CUDA_DEVICE long get_next (
          const long     origin,
          const long     current,
          const double3 &ray,
                double  &Z,
                double  &dZ      ) const;

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


#endif // __RAYPAIR_CUH_INCLUDED__
