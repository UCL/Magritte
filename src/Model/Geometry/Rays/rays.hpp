// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __RAYS_HPP_INCLUDED__
#define __RAYS_HPP_INCLUDED__


#include "Io/io.hpp"
#include "Tools/types.hpp"
#include "Model/parameters.hpp"


///  RAYS: data struct containing directional discretization info
/////////////////////////////////////////////////////////////////

struct Rays
{

  public:

      Double2 x;         ///< x component of direction vector
      Double2 y;         ///< y component of direction vector
      Double2 z;         ///< z component of direction vector

      Double2 weights;   ///< weights for angular integration

      //Double1 xr;         ///< x component of direction vector
      //Double1 yr;         ///< y component of direction vector
      //Double1 zr;         ///< z component of direction vector

      Double1 Ix;        ///< x component of horizontal image axis
      Double1 Iy;        ///< y component of horizontal image axis

      Double1 Jx;        ///< x component of vertical image axis
      Double1 Jy;        ///< y component of vertical image axis
      Double1 Jz;        ///< z component of vertical image axis

      Long2 antipod;     ///< ray number of antipodal ray


      // Io
      int read (
          const Io         &io,
                Parameters &parameters);

      int write (
          const Io &io) const;


  private:

      long ncells;
      long nrays;


      int setup ();

      // Helper functions
      int setup_image_axis ();

      int setup_antipodal_rays ();


      static const string prefix;

};




#endif // __RAYS_HPP_INCLUDED__