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
  Double3 Ls;    ///< values
  Long3   nr;    ///< position indices

  Double1 Lss;   ///< linearized values
  Long1   nrs;   ///< linearized position indices

  Long1   size;


  inline void initialize (const Parameters &parameters, const size_t nrad_new);

  inline void clear ();

  inline void linearize_data ();

  inline int MPI_gather ();


  inline size_t index_first (const size_t p, const size_t k) const;
  inline size_t index_last  (const size_t p, const size_t k) const;


  inline double get_Ls (const size_t p, const size_t k, const size_t index) const;
  inline size_t get_nr (const size_t p, const size_t k, const size_t index) const;

  inline size_t get_size (const size_t p, const size_t k) const;

  inline void add_element (const size_t p, const size_t k, const size_t nr, const double Ls);


  private:

      size_t ncells;
      size_t nrad;

};


#include "lambda.tpp"


#endif // __LAMBDA_HPP_INCLUDED__
