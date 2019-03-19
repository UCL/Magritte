// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __RADIATION_HPP_INCLUDED__
#define __RADIATION_HPP_INCLUDED__


#include "Io/io.hpp"
#include "Tools/types.hpp"
#include "Tools/Parallel/wrap_Grid.hpp"
#include "Model/parameters.hpp"
#include "Model/Radiation/Frequencies/frequencies.hpp"
#include "Model/Radiation/Scattering/scattering.hpp"


///  Radiation: data structure for the radiation field
//////////////////////////////////////////////////////

struct Radiation
{

  public:

      Frequencies frequencies;

      Scattering  scattering;


      vReal2 u;         ///< u intensity              (r, index(p,f))
      vReal2 v;         ///< v intensity              (r, index(p,f))

      vReal2 U;         ///< U scattered intensity   (r, index(p,f))
      vReal2 V;         ///< V scattered intensity   (r, index(p,f))

      vReal1 J;         ///< (angular) mean intensity (index(p,f))
      vReal1 G;         ///< (angular) mean intensity (index(p,f))

      //vReal1 J_local;   ///< local approximation to J (index(p,f))

      vReal3 I_bdy;     ///< intensity at the boundary (r,b,f)


      int read (
          const Io         &io,
                Parameters &parameters);

      int write (
          const Io &io) const;




    inline long index (
        const long p,
        const long f  ) const;

    inline long index (
        const long p,
        const long f,
        const long m  ) const;

    inline vReal get_U (
        const long R,
        const long p,
        const long f   ) const;


    inline vReal get_V (
        const long R,
        const long p,
        const long f   ) const;

    inline vReal get_I_bdy (
        const long R,
        const long p,
        const long f       ) const;


#if (GRID_SIMD)

    inline double get_U (
        const long R,
        const long p,
        const long f,
        const long lane ) const;

    inline double get_V (
        const long R,
        const long p,
        const long f,
        const long lane ) const;

    inline double get_I_bdy (
        const long R,
        const long p,
        const long f,
        const long lane     ) const;

#endif

    inline void rescale_U_and_V (
        const vReal &freq_scaled,
        const long   R,
        const long   p,
              long  &notch,
              vReal &U_scaled,
              vReal &V_scaled   ) const;

    inline void rescale_I_bdy (
        const vReal &freq_scaled,
        const long   R,
        const long   p,
        const long   b,
              long  &notch,
              vReal &Ibdy_scaled) const;


  int calc_J_and_G ();

  int calc_U_and_V ();


  private:

      long ncells;                ///< number of cells
      long nrays;                 ///< number of rays
      long nrays_red;             ///< reduced number of rays
      long nfreqs_red;             ///< reduced number of frequencies
      long nboundary;             ///< number of boundary cells
      long n_off_diag;             ///< number of boundary cells

  //template <int Dimension, long Nrays>
  //int compute_mean_intensity                      (
  //    const CELLS <Dimension, Nrays> &cells,
  //    const TEMPERATURE              &temperature,
  //    const FREQUENCIES              &frequencies,
  //    const LINES                    &lines,
  //    const SCATTERING               &scattering  );

  //template <int Dimension, long Nrays>
  //int compute_images                              (
  //    const CELLS <Dimension, Nrays> &cells,
  //    const TEMPERATURE              &temperature,
  //    const FREQUENCIES              &frequencies,
  //    const LINES                    &lines,
  //    const SCATTERING               &scattering  );

  //template <int Dimension, long Nrays>
  //int compute_mean_intensity_and_images           (
  //    const CELLS <Dimension, Nrays> &cells,
  //    const TEMPERATURE              &temperature,
  //    const FREQUENCIES              &frequencies,
  //    const LINES                    &lines,
  //    const SCATTERING               &scattering  );

};


#include "radiation.tpp"


#endif // __RADIATION_HPP_INCLUDED__
