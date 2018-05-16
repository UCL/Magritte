
// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <math.h>
#include "interpolation.hpp"
#include <iostream>

///  interpolate: interpolate tabulated function for a given range
///  @param[in] *f: pointer to tabulated function values
///  @param[in] *x: pointer to tabulated argument values
///  @param[in] start: start point to look for interpolation
///  @param[in] stop: end point to look for interpolation
///  @param[in] value: function argument to which we interpolate
///  @return function f evaluated at value
//////////////////////////////////////////////////////////////////

double interpolate (double *f, double *x, long start, long stop, double value)
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

	return (f[n]-f[n-1]) / (x[n]-x[n-1]) * (value-x[n-1]) + f[n-1];
}		




///  interpolate: interpolate tabulated function for a given range
///  @param[in] *f: pointer to tabulated function values
///  @param[in] *x: pointer to tabulated argument values
///  @param[in] start: start point to look for interpolation
///  @param[in] stop: end point to look for interpolation
///  @param[in] value: function argument to which we interpolate
///  @return index of x table just above value
//////////////////////////////////////////////////////////////////

long search (double *x, long start, long stop, double value)
{
  for (long n = start; n < stop; n++)
  {
  	if (value < x[n]) return n;
	}
}
