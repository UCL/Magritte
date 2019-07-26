// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __CELLS_HPP_INCLUDED__
#define __CELLS_HPP_INCLUDED__


#include "Io/io.hpp"
#include "Tools/types.hpp"
#include "Model/parameters.hpp"


///  CELLS: data structure containing all geometric data
////////////////////////////////////////////////////////

struct Cells
{

  public:

      Double1  x;          ///< x coordinate of cell center [m]
      Double1  y;          ///< y coordinate of cell center [m]
      Double1  z;          ///< z coordinate of cell center [m]

      Double1 vx;          ///< x component of velocity field (as fraction of C) [.]
      Double1 vy;          ///< y component of velocity field (as fraction of C) [.]
      Double1 vz;          ///< z component of velocity field (as fraction of C) [.]

      Long1 n_neighbors;   ///< number of neighbors of each cell
      Long2   neighbors;   ///< cell numbers of the neighbors of each cell


      // Io

      int read (
          const Io         &io,
                Parameters &parameters);

      int write (
          const Io &io) const;


  private:

      long ncells;                  ///< number of cells
      long ncameras;                ///< number of cameras
      long ncells_plus_ncameras;    ///< number of cells plus number of cameras

      static const string prefix;   ///< prefix to be used in io


};


#endif // __CELLS_HPP_INCLUDED__
