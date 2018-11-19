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
  const long START    = MPI_start (total);
  const long SUBTOTAL = MPI_length (total);

  return START + OMP_start (SUBTOTAL);
}


inline long HYBRID_stop (const long total)
{
  const long START    = MPI_start (total);
  const long SUBTOTAL = MPI_length (total);

  return START + OMP_stop (SUBTOTAL);
}


#endif // __HYBRID_TOOLS_HPP_INCLUDED__
