// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __CELLS_HPP_INCLUDED__
#define __CELLS_HPP_INCLUDED__


#include <vector>
using namespace std;

#include "types.hpp"
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

  const long ncells;                    ///< number of cells

  const RAYS <Dimension, Nrays> rays;   ///< rays linking different cells


  Double1  x,  y,  z;                   ///< coordinates of cell center
  Double1 vx, vy, vz;                   ///< components of velocity field

  Bool1 boundary;                       ///< true if boundary cell
  Bool1 mirror;                         ///< true if reflective boundary

	long nboundary;                       ///< number of boundary cells
	Long1 bdy_to_cell_nr;                 ///< boundary number of cell
	Long1 cell_to_bdy_nr;                 ///< cell number of boundary

  Long1 n_neighbors;                    ///< number of neighbors
  Long2   neighbors;                    ///< cell numbers of neighors

  Long1 id;                             ///< cell nr of corresp. cell in other grid
  Bool1 removed;                        ///< true when cell is removed


  CELLS (const long number_of_cells,
			   const string n_neighbors_file);   ///< Constructor


  int read (const string cells_file,
			      const string neighbors_file,
						const string boundary_file);


  long next (const long origin, const long ray, const long current,
             double& Z, double& dZ) const;

  long on_ray (const long origin, const long ray,
               long *cellNrs, double *dZs) const;


  double relative_velocity (const long origin, const long r, long current) const;


};


#include "cells.tpp"   // Implementation of template functions


#endif // __CELLS_HPP_INCLUDED__
