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
  //cout << a << " " << b << endl;

  return fabs ((a-b) / (a+b));
}


# if (GRID_SIMD)


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

    error = fabs (error);
  
  
  return error;

}


# endif




///  print: print a vectorized value
///    a: vector to print
////////////////////////////////////

inline int print (vReal a)
{

# if (GRID_SIMD)
    for (int lane = 0; lane < n_simd_lanes; lane++)
    {
      cout << a.getlane(lane) << "\t";
    }
# else
    cout << a;
# endif

  cout << endl;

  return (0);

}
