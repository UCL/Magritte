// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __HYBRID_TOOLS_HPP_INCLUDED__
#define __HYBRID_TOOLS_HPP_INCLUDED__


#define HYBRID_PARALLEL true

#if (HYBRID_PARALLEL)
#include "ompTools.hpp"
#include "mpiTools.hpp"
#endif


// IMPORTANT NOTE: HYBRID loops are also OMP loops and
// hence should be define in an OMP parallel region

inline long HYBRID_start (const long total)
{
  const long start    = MPI_start (total);
  const long subtotal = MPI_length (total);

  return start + OMP_start (subtotal);
}


inline long HYBRID_stop (const long total)
{
  const long start    = MPI_start (total);
  const long subtotal = MPI_length (total);

  return start + OMP_stop (subtotal);
}


#endif // __HYBRID_TOOLS_HPP_INCLUDED__
