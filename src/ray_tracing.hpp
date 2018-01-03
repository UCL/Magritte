// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __RAY_TRACING_HPP_INCLUDED__
#define __RAY_TRACING_HPP_INCLUDED__


#include "../parameters.hpp"
#include "Magritte_config.hpp"
#include "declarations.hpp"


#if (!CELL_BASED)


  // find_evalpoints: create evaluation points for each ray from this cell
  // ---------------------------------------------------------------------

  int find_evalpoints (CELL *cell, EVALPOINT *evalpoint, long *key, long *raytot, long *cum_raytot, long gridp);


  // get_velocities: get velocity of evaluation point w. r. t. originating cell
  // --------------------------------------------------------------------------

  int get_velocities (CELL *cell, EVALPOINT *evalpoint, long *key, long *raytot, long *cum_raytot, long gridp, long *first_velo);


#else


  // find_neighbors: find neighboring cells for each cell
  // ----------------------------------------------------

  int find_neighbors (long ncells, CELL *cell);


  // next_cell: find number of next cell on ray and its distance along ray
  // ---------------------------------------------------------------------

  long next_cell (long ncells, CELL *cell, long origin, long ray, double *Z, long current, double *dZ);


  // relative_velocity: get relative velocity of (cell) current w.r.t. (cell) origin along ray
  // -----------------------------------------------------------------------------------------

  double relative_velocity (long ncells, CELL *cell, long origin, long ray, long current);


#endif


#endif // __RAY_TRACING_HPP_INCLUDED__
