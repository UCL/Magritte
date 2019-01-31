// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __CELLS_HPP_INCLUDED__
#define __CELLS_HPP_INCLUDED__


#include "io.hpp"
#include "types.hpp"
#include "rays.hpp"


///  CELLS: data structure containing all geometric data
////////////////////////////////////////////////////////

struct Cells
{

  public:

      long ncells;              ///< number of cells
      long nboundary;           ///< number of boundary cells
      long nrays;               ///< number of rays

      Rays rays;                ///< rays discretizing the unit sphere

      Double1  x,  y,  z;       ///< [m] coordinates of cell center
      Double1 vx, vy, vz;       ///< [.] components of velocity field (as fraction of C)

      Bool1 boundary;           ///< true if boundary cell
      //Bool1 mirror;           ///< true if reflective boundary


      Long1 boundary2cell_nr;   ///< boundary number of cell
      Long1 cell2boundary_nr;   ///< cell number of boundary

      Long1 n_neighbors;        ///< number of neighbors
      Long2   neighbors;        ///< cell numbers of neighors


      // Constructor
      Cells (
          const Io &io);


      // Writer for output
      int write (
          const Io &io) const;


      // Inlined functions
      inline long next (
          const long    origin,
          const long    ray,
          const long    current,
                double &Z,
                double &dZ      ) const;

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


  private:

      int allocate ();

      int initialise ();

      int read (
          const Io &io);

      int setup ();

};


#include "../src/cells.tpp"


#endif // __CELLS_HPP_INCLUDED__
