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

      Double1  x,  y,  z;       ///< [m] coordinates of cell center
      Double1 vx, vy, vz;       ///< [.] components of velocity field (as fraction of C)

      Long1 n_neighbors;        ///< number of neighbors
      Long2   neighbors;        ///< cell numbers of neighors


      // Io
      int read (
          const Io         &io,
                Parameters &parameters);

      int write (
          const Io &io) const;


  private:

      long ncells;              ///< number of cells

      static const string prefix;


};


#endif // __CELLS_HPP_INCLUDED__
