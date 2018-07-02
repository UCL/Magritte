
// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <math.h>
#include <vector>
using namespace std;

#include "interpolation.hpp"


///  interpolate: interpolate tabulated function for a given range
///  @param[in] f: vector of tabulated function values
///  @param[in] x: vector of tabulated argument values
///  @param[in] start: start point to look for interpolation
///  @param[in] stop: end point to look for interpolation
///  @param[in] value: function argument to which we interpolate
///  @return function f evaluated at value
//////////////////////////////////////////////////////////////////

double interpolate (vector<double> f, vector<double> x, long start, long stop, double value)
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

	return interpolation_1 (x[n-1], f[n-1], x[n], f[n], value);
}		




///  interpolate: interpolate tabulated function for a given range
///  @param[in] x: vector of tabulated argument values
///  @param[in] start: start point to look for interpolation
///  @param[in] stop: end point to look for interpolation
///  @param[in] value: function argument to which we interpolate
///  @return index of x table just above value
//////////////////////////////////////////////////////////////////

long search (vector<double> x, long start, long stop, double value)
{
  for (long n = start; n < stop; n++)
  {
  	if (value < x[n]) return n;
	}

	return stop;
}




///  resample: resample function at x_new values
///  ASSUMING x_new preserves the order of x
///    @param[in] x: vector containing function arguments
///    @param[in] f: vector containing function values
///    @param[in] start: start point to look for interpolation
///    @param[in] stop: end point to look for interpolation
///    @param[in] x_new: vector containing new function arguments
///    @param[out] f_new: function values evaluated at new arguments
/////////////////////////////////////////////////////////////////////

int resample (vector<double> x, vector<double> f, 
		          const long start, const long stop,
	           	vector<double> x_new, vector<double>& f_new)
{

	long id     = start;
	long id_new = start;

	while ( (x_new[id_new] < x[start]) && (id_new < stop) )
	{
		f_new[id_new] = f[start];
		id_new++;
	}

	while ( (x_new[id_new] < x[stop-1]) && (id_new < stop) )
	{
  	while (x[id] < x_new[id_new])	id++;

    f_new[id_new] = interpolation_1 (x[id], f[id], x[id-1], f[id-1], x_new[id_new]);
		id_new++;
	}

	while (id_new < stop)
	{
		f_new[id_new] = f[stop-1];
		id_new++;
	}


	return (0);

}




///  interpolation_1: linear interpolation of f(x) in interval [x1, x2]
///    @param[in] x1: function argument 1
///    @param[in] f1: f(x1)
///    @param[in] x2: function argument 2
///    @param[in] f2: f(x2)
///    @param[in] x: value at which the function has to be interpolated
///    @return interpolated function value f(x)
///////////////////////////////////////////////////////////////////////

inline double interpolation_1 (double x1, double f1, 
		                           double x2, double f2, double x)
{
	return (f2-f1)/(x2-x1) * (x-x1) + f1;
}