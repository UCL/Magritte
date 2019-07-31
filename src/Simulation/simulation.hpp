// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __SIMULATION_HPP_INCLUDED__
#define __SIMULATION_HPP_INCLUDED__


#include "Io/io.hpp"
#include "Tools/types.hpp"
#include "Tools/timer.hpp"
#include "Model/model.hpp"
#include "Image/image.hpp"
#include "Raypair/raypair.hpp"


///  Simulation:
////////////////

struct Simulation : public Model
{


  Double1 error_max;
  Double1 error_mean;

  int compute_spectral_discretisation ();


  // In sim_radiation.cpp

  int compute_boundary_intensities ();

  int compute_radiation_field ();

  inline double get_dshift_max (
        const long o           );

  inline void setup (
      const long     R,
      const long     origin,
      const long     f,
            RayData &rayData_ar,
            RayData &rayData_r,
            RayPair &rayPair    ) const;

  inline void get_eta_and_chi (
      const vReal &freq_scaled,
      const long   p,
            long  &lnotch,
            vReal &eta,
            vReal &chi         ) const;

  int compute_and_write_image (
        const Io  &io,
        const long r          );


  // In sim_lines.cpp

  int compute_LTE_level_populations ();

  int compute_level_populations (
      const Io &io              );

  int compute_level_populations_opts (
      const Io   &io,
      const bool  use_Ng_acceleration,
      const long  max_niterations     );

  void calc_Jeff ();


};


#endif // __SIMULATION_HPP_INCLUDED__
