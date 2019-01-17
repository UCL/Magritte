// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __CELLS_HPP_INCLUDED__
#define __CELLS_HPP_INCLUDED__


#include "types.hpp"
#include "rays.hpp"


///  CELLS: class (template) containing all geometric data and functions.
///  - ASSUMING Ncells is not known at compile time!
///    @param Dimension: spacial dimension of grid
///    @param Nrays: number of rays oroginating from each cell
///    @param FixedNcells: true if number of cells is know at compile time
///    @param Ncells: number of cells (fixed in this case)
//////////////////////////////////////////////////////////////////////////

struct Cells
{

  const long ncells;                    ///< number of cells
  const long nrays;                     ///< number of rays

  Rays rays;                            ///< rays discretizing the unit sphere

  Double1  x,  y,  z;                   ///< coordinates of cell center
  Double1 vx, vy, vz;                   ///< components of velocity field (as fraction of C)

  Bool1 boundary;                       ///< true if boundary cell
  //Bool1 mirror;                         ///< true if reflective boundary

  long  nboundary;                      ///< number of boundary cells

  Long1 boundary2cell_nr;               ///< boundary number of cell
  Long1 cell2boundary_nr;               ///< cell number of boundary

  Long1 n_neighbors;                    ///< number of neighbors
  Long2   neighbors;                    ///< cell numbers of neighors


  // Constructor
  Cells (
      const Input input);   ///< Constructor


  // Setup and I/O
  int read (
      const Input input);

  int write (
      const string output_folder);

  int setup ();


  // Inlined functions
  inline long next (
      const long    origin,
      const long    ray,
      const long    current,
            double &Z,
            double &dZ      ) const;

  // inline long on_ray        (
  //     const long    origin,
  //     const long    ray,
  //           long   *cellNrs,
  //           double *dZs     ) const;

  inline double doppler_shift (
      const long origin,
      const long r,
      const long current      ) const;

  inline double x_projected (
      const long p,
      const long r          ) const;

  inline double y_projected (
      const long p,
      const long r          ) const;

};


#include "../src/cells.tpp"


#endif // __CELLS_HPP_INCLUDED__
