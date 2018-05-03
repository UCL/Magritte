// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __CELLS_F_HPP_INCLUDED__
#define __CELLS_F_HPP_INCLUDED__

#include <string>

#include "declarations.hpp"
#include "rays.hpp"


///  CELLS: class (template) containing all geometric data and functions.
///  - ASSUMING Ncells is known at compile time!
///    @param Dimension: spacial dimension of grid
///    @param Nrays: number of rays oroginating from each cell
///    @param FixedNcells: true if number of cells is know at compile time
///    @param Ncells: number of cells (fixed in this case)
//////////////////////////////////////////////////////////////////////////

template <int Dimension, long Nrays, long Ncells>
struct CELLS
{

  long ncells = Ncells;                 ///< number of cells

  const RAYS <Dimension, Nrays> rays;   ///< rays linking different cells


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


  int initialize ();                                                        ///< Initialize members

  long next (long origin, long ray, long current, double *Z, double *dZ);   ///< Next cell on ray

};


#include "cells_f.tpp"   // Implementation of template functions


#endif // __CELLS_F_HPP_INCLUDED__
