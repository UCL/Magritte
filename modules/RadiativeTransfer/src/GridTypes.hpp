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

  // When Grid is used

  #include <Grid.h>

  typedef Grid::vRealD vReal;

	const int n_simd_lanes = vReal :: Nsimd();

  typedef vector<vReal, Grid::alignedAllocator<vReal>> vReal1;

#else

  // When Grid is not used, use regular doubles

  typedef double vReal;

	const int n_simd_lanes = 1

  typedef vector<vReal> vReal1;

#endif


// Define tensors (types with more indices)

typedef vector<vReal1> vReal2;
typedef vector<vReal2> vReal3;
typedef vector<vReal3> vReal4;
typedef vector<vReal4> vReal5;


// Constant vector blocks

const vReal vZero = 0.0;
const vReal vOne  = 1.0;


// Define corresponding MPI type



#endif // __GRID_TYPES_HPP_INCLUDED__
