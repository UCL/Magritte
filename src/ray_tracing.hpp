// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __RAY_TRACING_HPP_INCLUDED__
#define __RAY_TRACING_HPP_INCLUDED__

#include "declarations.hpp"


// find_neighbors: find neighboring cells for each cell
// ----------------------------------------------------

int find_neighbors (long ncells, CELLS *cells, HEALPIXVECTORS healpixvectors);


// next_cell: find number of next cell on ray and its distance along ray
// ---------------------------------------------------------------------

long next_cell (long ncells, CELLS *cells, HEALPIXVECTORS healpixvectors,
                long origin, long ray, double *Z, long current, double *dZ);


// find_endpoints: find endpoint cells for each cell
// -------------------------------------------------

int find_endpoints (long ncells, CELLS *cells, HEALPIXVECTORS healpixvectors);


// previous_cell: find number of previous cell on ray and its distance along ray
// -----------------------------------------------------------------------------

long previous_cell (long ncells, CELLS *cells, HEALPIXVECTORS healpixvectors,
                    long origin, long ray, double *Z, long current, double *dZ);


// relative_velocity: get relative velocity of (cell) current w.r.t. (cell) origin along ray
// -----------------------------------------------------------------------------------------

double relative_velocity (long ncells, CELLS *cells, HEALPIXVECTORS healpixvectors,
                          long origin, long ray, long current);


#endif // __RAY_TRACING_HPP_INCLUDED__
