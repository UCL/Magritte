// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __HEAPSORT_HPP_INCLUDED__
#define __HEAPSORT_HPP_INCLUDED__



long max (double *a, long n, long i, long j, long k);

int downheap (double *a, long *b, long n, long i);

int heapsort (double *a, long *b, long n);


#endif // __HEAPSORT_HPP_INCLUDED__
