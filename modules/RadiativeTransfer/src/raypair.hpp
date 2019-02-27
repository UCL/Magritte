// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __RAYPAIR_HPP_INCLUDED__
#define __RAYPAIR_HPP_INCLUDED__


#include "cells.hpp"
#include "lines.hpp"
#include "frequencies.hpp"
#include "temperature.hpp"
#include "scattering.hpp"
#include "GridTypes.hpp"
#include "raydata.hpp"


///  RAYPAIR: data structure for a pair of rays
///////////////////////////////////////////////

struct RAYPAIR
{

  public:

    long ndep;

    vReal1   Su;   // effective source for u along the ray
    vReal1   Sv;   // effective source for v along the ray
    vReal1 dtau;   // optical depth increment along the ray

    vReal1 Lambda;


    RAYPAIR                           (
        const long num_of_cells,
        const long num_of_freq_red,
        const long ray_nr,
        const long aray_nr,
        const long Ray_nr,
        const vReal2 &U_local,
        const vReal2 &V_local,
        const vReal3 &Ibdy_local,
        const Long1  &Cell2boundary_nr);


    template <int Dimension, long Nrays>
    inline void initialize                        (
        const CELLS<Dimension,Nrays> &cells,
        const TEMPERATURE            &temperature,
        const long                    o           );


    inline void setup                  (
        const FREQUENCIES &frequencies,
        const TEMPERATURE &temperature,
        const LINES       &lines,
        const SCATTERING  &scattering,
        const long         f           );


    inline void solve (void);

    inline void solve_ndep_is_1 (void);


    inline vReal get_u_at_origin (void);
    inline vReal get_v_at_origin (void);

    inline vReal get_Lambda_at_origin (void);

    inline vReal get_I_p (void);
    inline vReal get_I_m (void);


  private:

    const long ncells;      ///< number of cells
    const long ray;
    const long aray;
    const long Ray;

    long origin;

    RAYDATA raydata_r;
    RAYDATA raydata_ar;

    vReal1 term1;   // effective source for u along the ray
    vReal1 term2;   // effective source for v along the ray

    vReal1 A;       // A coefficient in Feautrier recursion relation
    vReal1 C;       // C coefficient in Feautrier recursion relation
    vReal1 F;       // helper variable from Rybicki & Hummer (1991)
    vReal1 G;       // helper variable from Rybicki & Hummer (1991)

    vReal Ibdy_0;
    vReal Ibdy_n;

    vReal chi_at_origin;


    inline void fill_ar                (
        const FREQUENCIES &frequencies,
        const TEMPERATURE &temperature,
        const LINES       &lines,
        const SCATTERING  &scattering,
        const long         f           );

    inline void fill_r                 (
        const FREQUENCIES &frequencies,
        const TEMPERATURE &temperature,
        const LINES       &lines,
        const SCATTERING  &scattering,
        const long         f           );

};


#include "raypair.tpp"


#endif // __RAYPAIR_HPP_INCLUDED__
