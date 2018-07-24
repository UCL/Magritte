// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __INTERPOLATION_HPP_INCLUDED__
#define __INTERPOLATION_HPP_INCLUDED__


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
	                         const long stop, const double value);




inline int search (vReal1& vec, long& notch, const double value);

///  interpolate: interpolate tabulated function for a given range
///  @param[in] x: vector of tabulated argument values
///  @param[in] start: start point to look for interpolation
///  @param[in] stop: end point to look for interpolation
///  @param[in] value: function argument to which we interpolate
///  @return index of x table just above value
//////////////////////////////////////////////////////////////////

inline long search (const Double1& x, const long start,
	                  const long stop, const double value);




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
//inline int resample (vector<double>& x, vector<double>& f,
//		                 const long start, const long stop,
//	           	       vector<double>& x_new, vector<double>& f_new);




///  interpolation_1: linear interpolation of f(x) in interval [x1, x2]
///    @param[in] x1: function argument 1
///    @param[in] f1: f(x1)
///    @param[in] x2: function argument 2
///    @param[in] f2: f(x2)
///    @param[in] x: value at which the function has to be interpolated
///    @return interpolated function value f(x)
///////////////////////////////////////////////////////////////////////

inline double interpolation_1 (const double x1, const double f1,
                               const double x2, const double f2, const double x);




///  interpolation_1: linear interpolation of f(x) in interval [x1, x2]
///    @param[in] x1: function argument 1
///    @param[in] f1: f(x1)
///    @param[in] x2: function argument 2
///    @param[in] f2: f(x2)
///    @param[in] x: value at which the function has to be interpolated
///    @return interpolated function value f(x)
///////////////////////////////////////////////////////////////////////

inline vReal interpolation_1 (const vReal x1, const vReal f1,
                              const vReal x2, const vReal f2, const vReal x);


#include "interpolation.cpp"


#endif //  __INTERPOLATION_HPP_INCLUDED__
