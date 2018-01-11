// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __SETUP_DEFINITIONS_HPP_INCLUDED__
#define __SETUP_DEFINITIONS_HPP_INCLUDED__


#include <string>

#include "../parameters.hpp"


#ifndef TOL
# define TOL 1.0E-9   // tolerance for antipodal rays
#endif

#ifndef PI
# define PI  3.141592653589793238462643383279502884197   // pi
#endif

#ifndef LSPECPAR
# define LSPECPAR(lspec,par)   ( (par) + cum_ncolpar[(lspec)] )                                    \
                   /* when first index is line producing species and second is collision partner */
#endif

#ifndef VINDEX
# define VINDEX(r,c)   ( (c) + (r)*3 )   // when second index is a 3-vector index
#endif


#endif // __SETUP_DEFINITIONS_HPP_INCLUDED__
