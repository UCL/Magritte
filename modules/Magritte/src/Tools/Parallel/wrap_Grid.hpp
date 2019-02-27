// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __WRAP_GRID_HPP_INCLUDED__
#define __WRAP_GRID_HPP_INCLUDED__

#include <vector>


#define GRID_SIMD false


#if (GRID_SIMD)

  // When Grid is used

  #include <Grid.h>

  typedef Grid::vRealD vReal;

  const int n_simd_lanes = vReal::Nsimd();

  typedef std::vector<vReal, Grid::alignedAllocator<vReal>> vReal1;

#else

  // When Grid is not used, use regular doubles

  typedef double vReal;

  const int n_simd_lanes = 1;

  typedef std::vector<vReal> vReal1;

#endif


// Define tensors (types with more indices)

typedef std::vector<vReal1> vReal2;
typedef std::vector<vReal2> vReal3;
typedef std::vector<vReal3> vReal4;
typedef std::vector<vReal4> vReal5;


// Constant vector blocks

const vReal vZero = 0.0;
const vReal vOne  = 1.0;


// Helper functions

inline long reduced  (
    const long number )
{
  return (number + n_simd_lanes - 1) / n_simd_lanes;
}

inline int laneNr (
    const long index )
{
  return index % n_simd_lanes;
}

inline long newIndex (
    const long index )
{
  return index / n_simd_lanes;
}



#define GRID_FOR_ALL_LANES(lane) \
    for (int lane = 0; lane < n_simd_lanes; lane++)


inline double firstLane (
    const vReal vec     )

#if (GRID_SIMD)
{
  return vec.getlane (0);
}
#else
{
  return vec;
}
#endif


inline double lastLane (
    const vReal vec    )

#if (GRID_SIMD)
{
  return vec.getlane (n_simd_lanes-1);
}
#else
{
  return vec;
}
#endif




///  vExp: exponential function for vReal types
///  !!! Only good for positive exponents !!!
///    @param[in] x: exponent
///    @return exponential of x
/////////////////////////////////////////////////

inline vReal vExp (const vReal x)

#if (GRID_SIMD)

{

  const int n = 25;

  vReal result = 1.0;

  for (int i = n; i > 1; i--)
  {
    const double factor = 1.0 / i;   // INEFFICIENT -> STORE IN LIST
    const vReal vFactor = factor;

    result = vOne + x*result*vFactor;
  }

  result = vOne + x*result;


  return result;

}

#else

{
  return exp (x);
}

#endif



///  vExpMinus: exponential function for vReal types
///    @param[in] x: exponent
///    @return exponential of minus x
/////////////////////////////////////////////////

inline vReal vExpMinus (const vReal x)
{
  return 1.0 / vExp (x);
}




///  vExpm1: exponential minus 1.0 function for vReal types
///    @param[in] x: exponent
///    @return exponential minus 1.0 of x
///////////////////////////////////////////////////////////

inline vReal vExpm1 (const vReal x)

#if (GRID_SIMD)

{

  const int n = 30;

  vReal result = 1.0;

  for (int i = n; i > 1; i--)
  {
    const double factor = 1.0 / i;
    const vReal vFactor = factor;

    result = vOne + x*result*vFactor;
  }

  result = x*result;


  return result;

}

#else

{
  return expm1 (x);
}

#endif

#endif // __WRAP_GRID_HPP_INCLUDED__
