// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __CELLS_HPP_INCLUDED__
#define __CELLS_HPP_INCLUDED__

#include <string>

#include "declarations.hpp"
#include "rays.hpp"


///  CELLS: class (template) containing all geometric data and functions.
///  - ASSUMING Ncells is not known at compile time!
///    @param Dimension: spacial dimension of grid
///    @param Nrays: number of rays oroginating from each cell
///    @param FixedNcells: true if number of cells is know at compile time
///    @param Ncells: number of cells (fixed in this case)
//////////////////////////////////////////////////////////////////////////

template <int Dimension, long Nrays>
struct CELLS
{

  long ncells;                          ///< number of cells

  const RAYS <Dimension, Nrays> rays;   ///< rays linking different cells


  // Allocate on heap (in constructor)

  double  *x,  *y,  *z;           ///< coordinates of cell center
  double *vx, *vy, *vz;           ///< components of velocity field

  bool *boundary;                 ///< true if boundary cell
  bool *mirror;                   ///< true if reflective boundary

  long *neighbor;                 ///< cell numbers of neighors
  long *n_neighbors;              ///< number of neighbors

  long *id;                       ///< cell nr of associated cell in other grid
  bool *removed;                  ///< true when cell is removed


  CELLS (long number_of_cells);   ///< Constructor

  ~CELLS ();                      ///< Destructor

  int initialize ();                                                        ///< Initializemembers

  long next (long origin, long ray, long current, double *Z, double *dZ);   ///< Next cell on ray

  double relative_velocity (long origin, long r, long current);             ///< relative velocity

};


#include "cells.tpp"   // Implementation of template functions


#endif // __CELLS_HPP_INCLUDED__
