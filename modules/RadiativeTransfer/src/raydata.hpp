// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __RAYDATA_HPP_INCLUDED__
#define __RAYDATA_HPP_INCLUDED__


#include "cells.hpp"
#include "lines.hpp"
#include "frequencies.hpp"
#include "temperature.hpp"
#include "scattering.hpp"
#include "GridTypes.hpp"


///  RAYDATA: data structure for the data along a ray
/////////////////////////////////////////////////////

struct RAYDATA
{

  const long ncells;      ///< number of cells
  const long nfreq_red;
  const long ray;         ///< (global) index of ray
  const long Ray;         ///< (local) index of ray
  
  Long1  cellNrs;
  Long1    notch;
  Long1   lnotch;
  Double1 shifts;   // indicates where we are in frequency space
  Double1    dZs;

  vReal2 U, V;
  vReal3 boundary_intensity;
  Long1  cell2boundary_nr;

  vReal term1_n, term2_n, chi_n, eta_n;   // next values
  vReal term1_c, term2_c, chi_c;          // current values

  vReal Ibdy_scaled;

  vReal dtau, inverse_dtau;


  long origin;               ///< cell nr of origin
  long n = 0;                ///< Number of (projected) cells on this ray
  

  RAYDATA                           (
      const long num_of_cells,
      const long num_of_freq_red,
      const long ray_nr,
      const long Ray_nr,
      const vReal2 &U_local,
      const vReal2 &V_local,
      const vReal3 &Ibdy_local,
      const Long1  &Cell2boundary_nr);


  template <int Dimension, long Nrays>
  inline void initialize                   (
      const CELLS<Dimension, Nrays> &cells,
      const long                     o     );


  inline void set_current_to_origin  (
      const FREQUENCIES &frequencies,
      const TEMPERATURE &temperature,
      const LINES       &lines,
      const SCATTERING  &scattering,
      const long         f           );

  inline void set_current_to_origin_bdy (
      const FREQUENCIES &frequencies,
      const TEMPERATURE &temperature,
      const LINES       &lines,
      const SCATTERING  &scattering,
      const long         f               );

  inline void compute_next           (
      const FREQUENCIES &frequencies,
      const TEMPERATURE &temperature,
      const LINES       &lines,
      const SCATTERING  &scattering,
      const long         f,
      const long         q           );

  inline void compute_next_bdy       (
      const FREQUENCIES &frequencies,
      const TEMPERATURE &temperature,
      const LINES       &lines,
      const SCATTERING  &scattering,
      const long         f           );

  inline vReal get_Su_r  (void) const;
  inline vReal get_Sv_r  (void) const;
  inline vReal get_Su_ar (void) const;
  inline vReal get_Sv_ar (void) const;

  inline vReal get_boundary_term_Su_r  (void) const;
  inline vReal get_boundary_term_Sv_r  (void) const;
  inline vReal get_boundary_term_Su_ar (void) const;
  inline vReal get_boundary_term_Sv_ar (void) const;

  inline void set_current_to_next (void);


  private:
  
    inline void compute_next_eta_and_chi (
      const FREQUENCIES &frequencies,
      const TEMPERATURE &temperature,
      const LINES       &lines,
      const SCATTERING  &scattering,
      const vReal        freq_scaled,
      const long         q            );

  inline void compute_next_terms_and_dtau (
      const vReal U_scaled,
      const vReal V_scaled,
      const long  q                       );


  
  inline void rescale_U_and_V (
      const FREQUENCIES &frequencies,
      const long         p,
      const long         R,
            long        &notch,
      const vReal       &freq_scaled,
            vReal       &U_scaled,
            vReal       &V_scaled    );

  inline void rescale_U_and_V_and_bdy_I (
      const FREQUENCIES &frequencies,
      const long         p,
      const long         R,
            long        &notch,
      const vReal       &freq_scaled,
            vReal       &U_scaled,
            vReal       &V_scaled,
            vReal       &Ibdy_scaled    );
      
  inline long index (
      const long p,
      const long f  ) const;

};


#include "raydata.tpp"


#endif // __RAYDATA_HPP_INCLUDED__
