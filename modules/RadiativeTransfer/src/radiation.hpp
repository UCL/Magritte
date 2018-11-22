// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __RADIATION_HPP_INCLUDED__
#define __RADIATION_HPP_INCLUDED__


#include "cells.hpp"
#include "lines.hpp"
#include "GridTypes.hpp"
#include "temperature.hpp"
#include "frequencies.hpp"
#include "scattering.hpp"


///  RADIATION: data structure for the radiation field
//////////////////////////////////////////////////////

struct RADIATION
{

  const long ncells;          ///< number of cells
  const long nrays;           ///< number of rays
  const long nrays_red;       ///< reduced number of rays
  const long nfreq_red;       ///< reduced number of frequencies
  const long nboundary;       ///< number of boundary cells
  
  vReal2 u;                   ///< u intensity           (r, index(p,f))
  vReal2 v;                   ///< v intensity           (r, index(p,f))
  
  vReal2 U;                   ///< U scattered intensity (r, index(p,f))
  vReal2 V;                   ///< V scattered intensity (r, index(p,f))
  
  vReal1 J;                   ///< (angular) mean intensity (index(p,f))
  vReal1 G;                   ///< (angular) mean intensity (index(p,f))

  vReal3 boundary_intensity;   ///< intensity at the boundary (r,b,f)
  
  Long1 cell2boundary_nr;
  
  RADIATION                      (
      const long num_of_cells,
      const long num_of_rays,
      const long num_of_freq_red,
      const long num_of_bdycells );


  //int initialize ();

  int read                                (
      const string boundary_intensity_file);

  int write                               (
      const string boundary_intensity_file) const;

  inline long index (
      const long p,
      const long f  ) const;


  int calc_boundary_intensities           (
      const Long1       &Boundary2cell_nr,
      const Long1       &Cell2boundary_nr,
      const FREQUENCIES &frequencies      );


  inline int rescale_U_and_V         (
      const FREQUENCIES &frequencies,
      const long         p,
      const long         R,
            long        &notch,
      const vReal       &freq_scaled,
            vReal       &U_scaled,
            vReal       &V_scaled    ) const;

  inline int rescale_U_and_V_and_bdy_I (
      const FREQUENCIES &frequencies,
      const long         p,
      const long         R,
            long        &notch,
      const vReal       &freq_scaled,
            vReal       &U_scaled,
            vReal       &V_scaled,
            vReal       &Ibdy_scaled   ) const;

  int calc_J (void);

  int calc_U_and_V                (
      const SCATTERING& scattering);

  // Print

  int print           (
      const string tag) const;

  template <int Dimension, long Nrays>
  int compute_mean_intensity                      (
      const CELLS <Dimension, Nrays> &cells,
      const TEMPERATURE              &temperature,
      const FREQUENCIES              &frequencies,
      const LINES                    &lines,
      const SCATTERING               &scattering  );

  template <int Dimension, long Nrays>
  int compute_images                              (
      const CELLS <Dimension, Nrays> &cells,
      const TEMPERATURE              &temperature,
      const FREQUENCIES              &frequencies,
      const LINES                    &lines,
      const SCATTERING               &scattering  );

  template <int Dimension, long Nrays>
  int compute_mean_intensity_and_images           (
      const CELLS <Dimension, Nrays> &cells,
      const TEMPERATURE              &temperature,
      const FREQUENCIES              &frequencies,
      const LINES                    &lines,
      const SCATTERING               &scattering  );

};


#include "radiation.tpp"


#endif // __RADIATION_HPP_INCLUDED__
