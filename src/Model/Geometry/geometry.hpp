// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __GEOMETRY_HPP_INCLUDED__
#define __GEOMETRY_HPP_INCLUDED__

#include <exception>

#include "Io/io.hpp"
#include "Model/parameters.hpp"
#include "Model/Geometry/Cells/cells.hpp"
#include "Model/Geometry/Rays/rays.hpp"
#include "Model/Geometry/Boundary/boundary.hpp"
#include "Model/Geometry/raydata.hpp"


///  Frame of reference used in geometry computations
/////////////////////////////////////////////////////

enum Frame {CoMoving, Rest};




///  Geometry: data structure containing all geometric data
///////////////////////////////////////////////////////////

struct Geometry
{

public:

    Cells    cells;
    Rays     rays;
    Boundary boundary;

    bool spherical_symmetry   = false;
    bool adaptive_ray_tracing = false;
    long max_npoints_on_rays = -1;


    void read  (const Io &io, Parameters &parameters);
    void write (const Io &io                        );


    // Inlined functions
    template <Frame frame>
    inline RayData trace_ray (
        const long   origin,
        const long   ray,
        const double dshift_max) const;

    template <Frame frame>
    inline size_t get_npoints_on_ray (
          const size_t origin,
          const size_t ray,
          const double dshift_max  ) const;

    inline int set_data (
        const long     crt,
        const long     nxt,
        const double   shift_crt,
        const double   shift_nxt,
        const double   dZ_loc,
        const double   dshift_max,
              RayData &rayData    ) const;

    inline long get_next (
        const long    origin,
        const long    ray,
        const long    current,
              double &Z,
              double &dZ      ) const;

    template <Frame frame>
    inline double get_doppler_shift (
        const long  origin,
        const long  r,
        const long  current         ) const;

    inline int set_adaptive_rays (
        const size_t order_min,
        const size_t order_max,
        const size_t sample_size );


private:

    size_t nrays;   ///< number of rays

    inline long get_next_spherical_symmetry (
          const long  origin,
          const long  ray,
          const long  current,
          double     &Z,
          double     &dZ                    ) const;

    inline long get_next_general (
          const long  origin,
          const long  ray,
          const long  current,
          double     &Z,
          double     &dZ         ) const;

};


#include "geometry.tpp"


#endif // __GEOMETRY_HPP_INCLUDED__
