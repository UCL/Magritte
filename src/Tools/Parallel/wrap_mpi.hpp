// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __WRAP_MPI_HPP_INCLUDED__
#define __WRAP_MPI_HPP_INCLUDED__


#include <string>
using std::string;

#include "configure.hpp"


#if (MPI_PARALLEL)

  #include <mpi.h>

#else

  #define MPI_COMM_WORLD 1

  inline void MPI_Comm_size (int dummy, int *comm_size) {*comm_size = 1;}
  inline void MPI_Comm_rank (int dummy, int *comm_rank) {*comm_rank = 0;}

#endif


/// MPI_comm_size: return size of communicator
//////////////////////////////////////////////

inline size_t MPI_comm_size ()
{
  int comm_size;
  MPI_Comm_size (MPI_COMM_WORLD, &comm_size);

  return comm_size;
}




/// MPI_comm_rank: return rank of communicator
//////////////////////////////////////////////

inline size_t MPI_comm_rank ()
{
    int comm_rank;
    MPI_Comm_rank (MPI_COMM_WORLD, &comm_rank);

    return comm_rank;
}




/// MPI_comm_rank: return rank of communicator
//////////////////////////////////////////////

inline string str_MPI_comm_rank ()
{
    return std::to_string (MPI_comm_rank ());
}




/// The following abstractions simplify MPI parallel for loops.
/// The index space [0, total-1] is divided such that each process
/// has to do [MPI_start(total), MPI_stop(total)] of the indices.
/// NOTE: Only works when MPI is initialized!


///  MPI_start: first index for this process
///    @param[in] total: length of the range
////////////////////////////////////////////

inline size_t MPI_start (const size_t total)
{
    return (MPI_comm_rank() * total) / MPI_comm_size();
}




///  MPI_stop: last index for this process
///    @param[in] total: length of the range
////////////////////////////////////////////

inline size_t MPI_stop (const size_t total)
{
    return ((MPI_comm_rank()+1) * total) / MPI_comm_size();
}




///  MPI_length: number of indices there are for this process
///    @param[in] total: length of the range
/////////////////////////////////////////////////////////////

inline size_t MPI_length (const size_t total)
{
    return MPI_stop (total) - MPI_start (total);
}




///  MPI_PARALLEL_FOR: abstraction for an MPI loop with index ranging over total
///    @param[in] index: index of the for loop
///    @param[in] total: total range of the for loop
////////////////////////////////////////////////////////////////////////////////

#define MPI_PARALLEL_FOR(index, total) \
    for (size_t index = MPI_start (total); index < MPI_stop (total); index++)




#endif // __WRAP_MPI_HPP_INCLUDED__
