// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __LAMBDA_HPP_INCLUDED__
#define __LAMBDA_HPP_INCLUDED__


#include <vector>

#include "Tools/types.hpp"


///  Lambda: data structure for the Lambda oprator
//////////////////////////////////////////////////

//struct Lambda
//{
//
//  Double1 Ls;
//  Long1   nr;
//
//
//  inline void add_entry (
//      const double Ls,
//      const long   nr     );
//
//};
//
//
//typedef std::vector<Lambda>  Lambda1;
//typedef std::vector<Lambda1> Lambda2;




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




//Double1 get_Lss (Lambda2 lambda)
//{
//
//  Double1 Lss;
//
//  for (Lambda1 L1 : lambda)
//  {
//    for (Lambda L : L1)
//    {
//      Lss.insert (Lss.end(), L.Ls.begin(), L.Ls.end());
//    }
//  }
//
//
//  return Lss;
//
//}

#include "lambda.tpp"


#endif // __LAMBDA_HPP_INCLUDED__
