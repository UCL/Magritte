// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __IMAGE_HPP_INCLUDED__
#define __IMAGE_HPP_INCLUDED__


#include "Io/io.hpp"
#include "Tools/types.hpp"
#include "Model/parameters.hpp"
#include "Model/Geometry/geometry.hpp"


///  Image: data structure for the images
/////////////////////////////////////////

struct Image
{

  public:

      const long ray_nr;   ///< number of the ray to be imaged

      Double1 ImX;         ///< x coordinate of point in image
      Double1 ImY;         ///< y coordinate of point in image

      Double2 I_p;         ///< intensity out along ray (index(p,f))
      Double2 I_m;         ///< intensity out along ray (index(p,f))


      Image (
          const long        ray_nr,
          const Parameters &parameters);


      int write (
          const Io &io) const;


      int set_coordinates (
          const Geometry &geometry);


  private:

      const long ncells;     ///< number of cells
      const long ncameras;   ///< number of cameras
      const long nfreqs;     ///< number of frequencies

      static const string prefix;

};


#endif // __IMAGE_HPP_INCLUDED__
