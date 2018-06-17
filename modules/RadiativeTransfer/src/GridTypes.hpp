// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __GRID_TYPES_HPP_INCLUDED__
#define __GRID_TYPES_HPP_INCLUDED__

#include <vector>
using namespace std;

#include <Grid.h>


// Grid aligned vector types



// Grid vector block
typedef Grid::vRealD vDouble;


const vDouble vZero = 0.0;
const vDouble vOne  = 1.0;

// Number of vector (SIMD) lanes
const int n_vector_lanes = vDouble :: Nsimd();

// Full Grid vector
typedef vector<vDouble, Grid::alignedAllocator<vDouble>> vDouble1;

typedef vector<vDouble1> vDouble2;
typedef vector<vDouble2> vDouble3;
typedef vector<vDouble3> vDouble4;
typedef vector<vDouble4> vDouble5;


#endif // __GRID_TYPES_HPP_INCLUDED__
