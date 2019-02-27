// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __INTERPOLATION_HPP_INCLUDED__
#define __INTERPOLATION_HPP_INCLUDED__


#include "Tools/Parallel/wrap_Grid.hpp"
#include "Tools/types.hpp"


///  search: look up the index of a value in a list
///  @param[in] x: vector of tabulated argument values
///  @param[in] start: start point to look for value
///  @param[in] stop: end point to look for value
///  @param[in] value: value to search for
///  @return index of x table just above value
//////////////////////////////////////////////////////

inline long search (
    const Double1 &x,
    const double   value);




///  search_with_notch: linear search for value in ordered list vec
///    @param[in] vec: vectorized (and ordered) list in which to search value
///    @param[in/out] notch:
///    @param[in] value: the value we look for in vec
/////////////////////////////////////////////////////////////////////////////

inline int search_with_notch (
    const vReal1 &vec,
          long   &notch,
    const double  value);




///  interpolate_linear: linear interpolation of f(x) in interval [x1, x2]
///    @param[in] x1: function argument 1
///    @param[in] f1: f(x1)
///    @param[in] x2: function argument 2
///    @param[in] f2: f(x2)
///    @param[in] x: value at which the function has to be interpolated
///    @return interpolated function value f(x)
//////////////////////////////////////////////////////////////////////////

inline double interpolate_linear (
    const double x1,
    const double f1,
    const double x2,
    const double f2,
    const double x  );



#if (GRID_SIMD)

///  interpolate_linear: linear interpolation of f(x) in interval [x1, x2]
///    @param[in] x1: function argument 1
///    @param[in] f1: f(x1)
///    @param[in] x2: function argument 2
///    @param[in] f2: f(x2)
///    @param[in] x: value at which the function has to be interpolated
///    @return interpolated function value f(x)
//////////////////////////////////////////////////////////////////////////

inline vReal interpolate_linear (
    const vReal x1,
    const vReal f1,
    const vReal x2,
    const vReal f2,
    const vReal x  );

#endif


#include "interpolation.tpp"


#endif //  __INTERPOLATION_HPP_INCLUDED__
