// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __SIMULATION_HPP_INCLUDED__
#define __SIMULATION_HPP_INCLUDED__


#include "Io/io.hpp"
#include "Tools/types.hpp"
#include "Tools/timer.hpp"
#include "Tools/logger.hpp"
#include "Model/model.hpp"
#include "Image/image.hpp"
#include "Raypair/raypair.hpp"
//#include "Raypair/rayblock.hpp"

#if (GPU_ACCELERATION)
#   include <cuda_runtime.h>
#   include "Raypair/raypair.cuh"
#   include "Raypair/rayblock.cuh"
#endif


enum SpecDiscSetting {None, LineSet, ImageSet};


///  Simulation:
////////////////

struct Simulation : public Model
{
    Double2 dtaus;
    Double2 dZs;
    Double2 chis;
    Long2   pre;
    Long2   pos;

# if (GPU_ACCELERATION)
    int handleCudaError (cudaError_t error);
    int gpu_get_device_properties     ();
//    int gpu_compute_radiation_field   ();
    int gpu_compute_radiation_field_2 (const size_t nraypairs, const size_t gpuBlockSize, const double inverse_dtau_max);
# endif

    int cpu_compute_radiation_field_2 (const size_t nraypairs);


  Double1 error_max;
  Double1 error_mean;

  SpecDiscSetting specDiscSetting = None;


  //vReal tau_max = 10.0;

  int compute_spectral_discretisation       (void);
  int compute_spectral_discretisation_image (const double width);
  int compute_boundary_intensities          (void);
  int compute_boundary_intensities          (const Double1 &temperatures);
  int compute_radiation_field               (void);

  inline double get_dshift_max (
        const long o           ) const;

  inline void setup_using_scattering (
      const long     R,
      const long     origin,
      const long     f,
            RayData &rayData_ar,
            RayData &rayData_r,
            RayPair &rayPair    ) const;

  inline void setup (
      const long     R,
      const long     origin,
      const long     f,
            RayData &rayData_ar,
            RayData &rayData_r,
            RayPair &rayPair    ); // to append the dtaus etc const;

  inline void get_eta_and_chi (
      const vReal &freq_scaled,
      const long   p,
            long  &lnotch,
            vReal &eta,
            vReal &chi         ) const;

  inline double get_line_width (
        const long p,
        const long lindex      ) const;

  int compute_and_write_image (
        const Io  &io,
        const long r          );


  // In sim_lines.cpp

  int compute_LTE_level_populations ();

  long compute_level_populations (
      const Io &io               );

  long compute_level_populations (
      const Io   &io,
      const bool  use_Ng_acceleration,
      const long  max_niterations     );

  void calc_Jeff ();


  template <Frame frame>
  Long1 get_npoints_on_ray (
      const long r         ) const;

  template <Frame frame>
  long get_max_npoints_on_ray (
      const long r            ) const;

  template <Frame frame>
  Long2 get_npoints_on_rays () const;

  template <Frame frame>
  long get_max_npoints_on_rays ();


  inline void get_radiation_field_from_boundary (
      const long R,
      const long r,
      const long o                              );

};


#include "simulation.tpp"


#endif // __SIMULATION_HPP_INCLUDED__
