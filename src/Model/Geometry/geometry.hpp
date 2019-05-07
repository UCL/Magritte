// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __GEOMETRY_HPP_INCLUDED__
#define __GEOMETRY_HPP_INCLUDED__


#include "Io/io.hpp"
#include "Model/parameters.hpp"
#include "Model/Geometry/Cells/cells.hpp"
#include "Model/Geometry/Rays/rays.hpp"
#include "Model/Geometry/Boundary/boundary.hpp"
#include "Model/Geometry/raydata.hpp"


///  Geometry: data structure containing all geometric data
////////////////////////////////////////////////////////

struct Geometry
{

  public:

      Cells    cells;

      Rays     rays;

      Boundary boundary;


      int read (
          const Io         &io,
                Parameters &parameters);

      int write (
          const Io &io) const;


      // Inlined functions
      inline RayData trace_ray (
          const long   origin,
          const long   ray,
          const double dshift_max) const;

      inline void set_data (
          const long     crt,
          const long     nxt,
          const double   shift_crt,
          const double   shift_nxt,
          const double   dZ_loc,
          const double   dshift_max,
                RayData &rayData    ) const;

      inline long next (
          const long    origin,
          const long    ray,
          const long    current,
                double &Z,
                double &dZ      ) const;

      inline double doppler_shift (
          const long origin,
          const long r,
          const long current      ) const;


};


#include "geometry.tpp"


#endif // __GEOMETRY_HPP_INCLUDED__
