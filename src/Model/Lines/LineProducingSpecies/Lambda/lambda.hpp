// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __LAMBDA_HPP_INCLUDED__
#define __LAMBDA_HPP_INCLUDED__


#include <vector>

#include "Tools/types.hpp"
#include "Tools/Parallel/wrap_omp.hpp"
#include "Tools/Parallel/wrap_mpi.hpp"


struct Lambda
{

  Double3 Ls;
  Long3   nr;

  Double1 Lss;
  Long1   nrs;

  Long1   size;


  inline int initialize (
      const Parameters &parameters,
      const long        nrad_new   );

  inline int clear ();

  inline int linearize_data ();

  inline int MPI_gather ();


  inline long index_first (
      const long p,
      const long k        ) const;

  inline long index_last (
      const long p,
      const long k       ) const;


  inline double get_Ls (
      const long p,
      const long k,
      const long index ) const;

  inline long get_nr (
      const long p,
      const long k,
      const long index ) const;

  inline long get_size (
      const long p,
      const long k     ) const;


  inline void add_element (
      const long   p,
      const long   k,
      const long   nr,
      const double Ls     );


  private:

      long ncells;
      long nrad;


};


#include "lambda.tpp"


#endif // __LAMBDA_HPP_INCLUDED__
