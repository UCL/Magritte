// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <math.h>
#include <iostream>

#include "GridTypes.hpp"


///  relative_error: compute the relative difference between two values
///    @param[in] a: value 1
///    @paran[in] b: value 2
///    @return relative differencte between a and b
///////////////////////////////////////////////////////////////////////

inline double relative_error (const double a,
                              const double b )
{
  return fabs ((a-b) / (a+b));
}




///  relative_error: compute the relative difference between two vectors
///    @param[in] a: value 1
///    @paran[in] b: value 2
///    @return relative differencte between a and b
////////////////////////////////////////////////////////////////////////

inline vReal relative_error (const vReal a,
                             const vReal b )
{

  vReal error = (a-b) / (a+b);
  
  
  // Take absolute value of individual simd lanes
  
  for (int lane = 0; lane < n_simd_lanes; lane++)
  {
    error.putlane (fabs (error.getlane (lane)), lane);
  }
  
  
  return error;

}




///  print: print a vectorized value
///    a: vector to print
////////////////////////////////////

inline int print (vReal a)
{

  for (int lane = 0; lane < n_simd_lanes; lane++)
  {
    cout << a.getlane(lane) << "\t";
  }

  cout << endl;


  return (0);

}
