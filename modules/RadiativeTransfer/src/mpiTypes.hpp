// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __MPI_TYPES_HPP_INCLUDED__
#define __MPI_TYPES_HPP_INCLUDED__


#define MPI_PARALLEL true

#if (MPI_PARALLEL)
#include <mpi.h>
#endif

extern const int world_size;
extern const int world_rank;


#endif // __MPI_TYPES_HPP_INCLUDED__
