// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __MPI_TOOLS_HPP_INCLUDED__
#define __MPI_TOOLS_HPP_INCLUDED__


#define MPI_PARALLEL true

#if (MPI_PARALLEL)
#include <mpi.h>
#endif


inline int MPI_comm_size (void)
{
  int comm_size;
  MPI_Comm_size (MPI_COMM_WORLD, &comm_size);

  return comm_size;
}


inline int MPI_comm_rank (void)
{
  int comm_rank;
  MPI_Comm_rank (MPI_COMM_WORLD, &comm_rank);

  return comm_rank;
}




inline long MPI_start (const long total)
{
  return (MPI_comm_rank() * total) / MPI_comm_size();
}


inline long MPI_stop (const long total)
{
  return ((MPI_comm_rank()+1) * total) / MPI_comm_size();
}


inline long MPI_length (const long total)
{
  return MPI_stop (total) - MPI_start (total);
}



#endif // __MPI_TOOLS_HPP_INCLUDED__
