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

struct Lambda
{

  Double1 Ls;
  Long1   nr;

  inline void add_entry (
      const double Ls,
      const long   nr     );

};


typedef std::vector<Lambda>  Lambda1;
typedef std::vector<Lambda1> Lambda2;


#include "lambda.tpp"


#endif // __LAMBDA_HPP_INCLUDED__
