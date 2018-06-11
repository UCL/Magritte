// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __HEAPSORT_HPP_INCLUDED__
#define __HEAPSORT_HPP_INCLUDED__

#include <vector>
using namespace std;


long max (vector<double>& a, long n, long i, long j, long k);

int downheap (vector<double>& a, vector<long>& b, long n, long i);

int heapsort (vector<double>& a, vector<long>& b, long n);


#endif // __HEAPSORT_HPP_INCLUDED__
