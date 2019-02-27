// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __WRAP_HYBRID_HPP_INCLUDED__
#define __WRAP_HYBRID_HPP_INCLUDED__


#define HYBRID_PARALLEL true

#if (HYBRID_PARALLEL)
#include "wrap_omp.hpp"
#include "wrap_mpi.hpp"
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


//#define HYBRID_PARALLEL_FOR(index, total) \
//    _Pragma("omp parallel default(shared)") \
//    for (long index = HYBRID_start (total); index < HYBRID_stop (total); index++)

#define HYBRID_PARALLEL_FOR(index, total) \
    for (long index = 0; index < total; index++)


#endif // __WRAP_HYBRID_HPP_INCLUDED__
