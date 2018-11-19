// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __OMP_TOOLS_HPP_INCLUDED__
#define __OMP_TOOLS_HPP_INCLUDED__


#define OMP_PARALLEL true

#if (OMP_PARALLEL)
#include <omp.h>
#endif


// OMP loops

inline long OMP_start (const long total)
{
  return (omp_get_thread_num() * total) / omp_get_num_threads();
}


inline long OMP_stop (const long total)
{
  return ((omp_get_thread_num()+1) * total) / omp_get_num_threads();
}


#endif // __OMP_TOOLS_HPP_INCLUDED__
