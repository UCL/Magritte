// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __RAYS_HPP_INCLUDED__
#define __RAYS_HPP_INCLUDED__


#include "io.hpp"
#include "types.hpp"


///  RAYS: data struct containing directional discretization info
/////////////////////////////////////////////////////////////////

struct Rays
{

  public:

      long nrays;

      Double1 x;         ///< x component of direction vector
      Double1 y;         ///< y component of direction vector
      Double1 z;         ///< z component of direction vector

      Double1 Ix;        ///< x component of horizontal image axis
      Double1 Iy;        ///< y component of horizontal image axis

      Double1 Jx;        ///< x component of vertical image axis
      Double1 Jy;        ///< y component of vertical image axis
      Double1 Jz;        ///< z component of vertical image axis

      Long1 antipod;     ///< ray number of antipodal ray


      // Constructor
      Rays (
          const Io &io);


      // Writer for output
      int write (
          const Io &io) const;


  private:

      int allocate ();

      int read (
          const Io &io);

      int setup ();


      // Helper functions
      int setup_image_axis ();

      int setup_antipodal_rays ();

};




#endif // __RAYS_HPP_INCLUDED__
