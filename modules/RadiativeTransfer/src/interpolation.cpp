
// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <math.h>
#include <vector>
using namespace std;

#include "interpolation.hpp"
#include "GridTypes.hpp"
#include "types.hpp"

///  interpolate: interpolate tabulated function for a given range
///  @param[in] f: vector of tabulated function values
///  @param[in] x: vector of tabulated argument values
///  @param[in] start: start point to look for interpolation
///  @param[in] stop: end point to look for interpolation
///  @param[in] value: function argument to which we interpolate
///  @return function f evaluated at value
//////////////////////////////////////////////////////////////////

inline double interpolate (const Double1& f, const Double1& x, const long start,
                           const long stop, const double value)
{
  if (value < x[start])
	{
		return f[start];
	}

	if (value > x[stop-1])
	{
		return f[stop-1];
	}

  long n = search (x, start, stop, value);

	return interpolate_linear (x[n-1], f[n-1], x[n], f[n], value);
}




/////  interpolate: interpolate tabulated function for a given range
/////  @param[in] x: vector of tabulated argument values
/////  @param[in] start: start point to look for interpolation
/////  @param[in] stop: end point to look for interpolation
/////  @param[in] value: function argument to which we interpolate
/////  @return index of x table just above value
////////////////////////////////////////////////////////////////////
//
//inline long search (const Double1& x, const long start,
//                    const long stop, const double value)
//{
//  for (long n = start; n < stop; n++)
//  {
//  	if (value < x[n]) return n;
//	}
//
//	return stop;
//}




inline int search_with_notch (vReal1& vec, long& notch, const double value)
{
  long f    = notch / n_simd_lanes;
   int lane = notch % n_simd_lanes;

  while (f < vec.size())
  {
    if (value < vec[f].getlane(lane)) return (0);

    notch++;

    f    = notch / n_simd_lanes;
    lane = notch % n_simd_lanes;
  }

  return (1);

}




inline long search (const Double1& x, long start, long stop, const double value)
{
  while (stop > start)
  {
    const long middle = (stop - start) / 2;

    if (value > x[middle])
    {
      start = middle;
    }
    else
    {
      stop = middle;
    }
  }

  return stop;
}



//
/////  resample: resample function at x_new values
/////  ASSUMING x_new preserves the order of x
/////    @param[in] x: vector containing function arguments
/////    @param[in] f: vector containing function values
/////    @param[in] start: start point to look for interpolation
/////    @param[in] stop: end point to look for interpolation
/////    @param[in] x_new: vector containing new function arguments
/////    @param[out] f_new: function values evaluated at new arguments
///////////////////////////////////////////////////////////////////////
//
//int resample (vector<double>& x, vector<double>& f,
//		          const long start, const long stop,
//	           	vector<double>& x_new, vector<double>& f_new)
//{
//
//	long id     = start;
//	long id_new = start;
//
//	while ( (x_new[id_new] < x[start]) && (id_new < stop) )
//	{
//		f_new[id_new] = f[start];
//		id_new++;
//	}
//
//	while ( (x_new[id_new] < x[stop-1]) && (id_new < stop) )
//	{
//  	while (x[id] < x_new[id_new])	id++;
//
//    f_new[id_new] = interpolate_linear (x[id], f[id], x[id-1], f[id-1], x_new[id_new]);
//		id_new++;
//	}
//
//	while (id_new < stop)
//	{
//		f_new[id_new] = f[stop-1];
//		id_new++;
//	}
//
//
//	return (0);
//
//}




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
