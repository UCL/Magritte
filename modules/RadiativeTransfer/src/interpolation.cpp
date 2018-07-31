
// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <math.h>

#include "interpolation.hpp"
#include "GridTypes.hpp"
#include "types.hpp"


///  search: binary search for the index of a value right above a value in a list
///  @param[in] x: vector of tabulated argument values
///  @param[in] value: value to search for
///  @return index of x table just above value
/////////////////////////////////////////////////////////////////////////////////

inline long search (const Double1& x, const double value)
{

  long middle;

  long start = 0;
  long stop  = x.size()-1;


  if      (value >= x[stop])
  {
    return stop;
  }
  else if (value <= x[start])
  {
    return start;
  }


  while (stop > start+1)
  {
    const long middle = (stop + start) / 2;

    if      (value > x[middle])
    {
      start = middle;
    }
    else if (value < x[middle])
    {
      stop  = middle;
    }
  }


  return stop;

}




///  search_with_notch: linear search for value in ordered list vec
///    @param[in] vec: vectorized (and ordered) list in which to search value
///    @param[in/out] notch:
///    @param[in] value: the value we look for in vec
/////////////////////////////////////////////////////////////////////////////

inline int search_with_notch (vReal1& vec, long& notch, const double value)

#if (GRID_SIMD)

{

  long f    = notch / n_simd_lanes;
   int lane = notch % n_simd_lanes;

  while (f < vec.size())
  {
    if (value <= vec[f].getlane(lane)) return (0);

    notch++;

    f    = notch / n_simd_lanes;
    lane = notch % n_simd_lanes;
  }


  return (1);

}

#else

{

  while (notch < vec.size())
  {
    if (value <= vec[notch]) return (0);

    notch++;
  }

  return (1);

}

#endif




///  interpolate_linear: linear interpolation of f(x) in interval [x1, x2]
///    @param[in] x1: function argument 1
///    @param[in] f1: f(x1)
///    @param[in] x2: function argument 2
///    @param[in] f2: f(x2)
///    @param[in] x: value at which the function has to be interpolated
///    @return interpolated function value f(x)
//////////////////////////////////////////////////////////////////////////

inline double interpolate_linear (const double x1, const double f1,
                                  const double x2, const double f2, const double x)
{
	return (f2-f1)/(x2-x1) * (x-x1) + f1;
}




///  interpolate_linear: linear interpolation of f(x) in interval [x1, x2]
///    @param[in] x1: function argument 1
///    @param[in] f1: f(x1)
///    @param[in] x2: function argument 2
///    @param[in] f2: f(x2)
///    @param[in] x: value at which the function has to be interpolated
///    @return interpolated function value f(x)
//////////////////////////////////////////////////////////////////////////

inline vReal interpolate_linear (const vReal x1, const vReal f1,
                                 const vReal x2, const vReal f2, const vReal x)
{
	return (f2-f1)/(x2-x1) * (x-x1) + f1;
}
