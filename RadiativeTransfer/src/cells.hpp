// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __CELLS_HPP_INCLUDED__
#define __CELLS_HPP_INCLUDED__

#include <string>

#include "declarations.hpp"
#include "rays.hpp"


///  CELLS: class containing all geometric data and functions for Radiative Transfer
////////////////////////////////////////////////////////////////////////////////////

template <int Dimension, long Nrays, bool Fixed_Ncells, long Ncells>

class CELLS
{

  public:

    long ncells;   ///< number of cells

#   if (Fixed_Ncells)

      // Allocate on stack

      double  x[Ncells],  y[Ncells],  z[Ncells];   ///< coordinates of cell center
      double vx[Ncells], vy[Ncells], vz[Ncells];   ///< components of velocity field

      bool boundary[Ncells];                       ///< true if boundary cell
      bool   mirror[Ncells];                       ///< true if reflective boundary

      long endpoint[Ncells*Nrays];                 ///< cell numbers of ray endings
      double      Z[Ncells*Nrays];                 ///< distance from cell to boundary

      long    neighbor[Ncells*Nrays];              ///< cell numbers of neighors
      long n_neighbors[Ncells];                    ///< number of neighbors

      long      id[Ncells];                        ///< cell nr of associated cell in other grid
      bool removed[Ncells];                        ///< true when cell is removed

#   else

      // Allocate on heap (in constructor)

      double  *x,  *y,  *z;   ///< coordinates of cell center
      double *vx, *vy, *vz;   ///< components of velocity field

      bool *boundary;         ///< true if boundary cell
      bool *mirror;           ///< true if reflective boundary

      long   *endpoint;       ///< cell numbers of ray endings
      double *Z;              ///< distance from cell to boundary

      long *neighbor;         ///< cell numbers of neighors
      long *n_neighbors;      ///< number of neighbors

      long *id;               ///< cell nr of associated cell in other grid
      bool *removed;          ///< true when cell is removed

#   endif


    CELLS (long number_of_cells);   ///< Constructor

    ~CELLS ();                      ///< Destructor

    int initialize ();              ///< Initializer for members

    long next();                    ///< Next cell along a ray


  private:

    const RAYS <Dimension, Nrays> rays;   ///< rays linking different cells

};


#include "cells.tpp"   // Implementation of template functions


#endif // __CELLS_HPP_INCLUDED__
