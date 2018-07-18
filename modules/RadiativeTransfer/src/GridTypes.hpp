// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __GRID_TYPES_HPP_INCLUDED__
#define __GRID_TYPES_HPP_INCLUDED__

#include <vector>
using namespace std;


#define GRID_SIMD true

#if (GRID_SIMD)
#include <Grid.h>
#endif

// Grid aligned vector types


// Grid vector block
// define away the reference to double or float so it can be altered here.

#if (GRID_SIMD)
  typedef Grid::vRealD vReal;
#else
  typedef double vReal;
#endif

// Number of SIMD (vector) lanes
#if (GRID_SIMD)
	const int n_simd_lanes = vReal :: Nsimd();
#else
	const int n_simd_lanes = 1;
#endif


// Full Grid vector
#if (GRID_SIMD)
typedef vector<vReal, Grid::alignedAllocator<vReal>> vReal1;
#else
typedef vector<vReal>  vReal1;
#endif

typedef vector<vReal1> vReal2;
typedef vector<vReal2> vReal3;
typedef vector<vReal3> vReal4;
typedef vector<vReal4> vReal5;


// Constant vector blocks

const vReal vZero = 0.0;
const vReal vOne  = 1.0;


// Define corresponding MPI type



#endif // __GRID_TYPES_HPP_INCLUDED__
