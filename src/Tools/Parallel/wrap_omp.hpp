// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __WRAP_OMP_HPP_INCLUDED__
#define __WRAP_OMP_HPP_INCLUDED__


#if (_OPENMP)
  #define OMP_PARALLEL true
#else
  #define OMP_PARALLEL false
#endif

#if (OMP_PARALLEL)
  #include <omp.h>
#else
  inline long omp_get_thread_num  () {return 0;}
  inline long omp_get_num_threads () {return 1;}
#endif


// OMP loops

inline long OMP_start (
    const long total  )
{
  return (omp_get_thread_num() * total) / omp_get_num_threads();
}


inline long OMP_stop (
    const long total )
{
  return ((omp_get_thread_num()+1) * total) / omp_get_num_threads();
}

//#define OMP_PARALLEL_FOR(index, total) \
//    _Pragma("omp parallel default(shared)") \
//    for (long index = OMP_start (total); index < OMP_stop (total); index++)

#define OMP_PARALLEL_FOR(index, total) \
    for (long index = 0; index < total; index++)

//#define OMP_PARALLEL_FOR_WITH(index, total, on_each_thread) \
//    _Pragma("omp parallel default(shared)") \
//    on_each_thread \
//    for (long index = OMP_start (total); index < OMP_stop (total); index++)

#endif // __WRAP_OMP_HPP_INCLUDED__
