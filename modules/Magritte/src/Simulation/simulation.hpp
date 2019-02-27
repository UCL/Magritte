// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __SIMULATION_HPP_INCLUDED__
#define __SIMULATION_HPP_INCLUDED__


#include "Io/io.hpp"
#include "Tools/types.hpp"
#include "Model/model.hpp"
#include "Raypair/raypair.hpp"


///  Simulation:
//////////////////////////////////////////////////////////////////

struct Simulation : public Model
{

  Double1 error_max;
  Double1 error_mean;


  int compute_spectral_discretisation ();

  int compute_boundary_intensities ();

  int compute_LTE_level_populations ();

  int compute_radiation_field ();

  int compute_level_populations ();


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

int update_using_statistical_equilibrium ();

void calc_J_and_L_eff (
      const long p,
      const int  l,
      const long k    );

Eigen::MatrixXd get_transition_matrix (
      const long p,
      const long l                    );

};


#endif // __SIMULATION_HPP_INCLUDED__
