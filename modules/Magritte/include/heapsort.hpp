// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __HEAPSORT_HPP_INCLUDED__
#define __HEAPSORT_HPP_INCLUDED__


#include "types.hpp"


long max (
    const Double1 &a,
    const long     n,
    const long     i,
    const long     j,
    const long     k );

int downheap (
    const Double1 &a,
    const Long1   &b,
    const long     n,
          long     i );

int heapsort (
    Double1 &a,
    Long1   &b );


#endif // __HEAPSORT_HPP_INCLUDED__
