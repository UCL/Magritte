// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __CAMERAS_HPP_INCLUDED__
#define __CAMERAS_HPP_INCLUDED__


#include "Io/io.hpp"
#include "Tools/types.hpp"
#include "Model/parameters.hpp"


///  Cameras: data structure containing camera data
///////////////////////////////////////////////////

struct Cameras
{

  public:

      long  ray_nr;           ///< number of the ray to be imaged

      Long1 camera2cell_nr;   ///< boundary number of cell


      // Io
      int read (
          const Io         &io,
                Parameters &parameters);

      int write (
          const Io &io) const;


  private:

      long ncameras;           ///< number of boundary cells

      static const string prefix;

};


#endif // __CAMERAS_HPP_INCLUDED__
