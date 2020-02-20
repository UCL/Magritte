// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __WRAP_GRID_HPP_INCLUDED__
#define __WRAP_GRID_HPP_INCLUDED__


#include "Tools/types.hpp"
#include "configure.hpp"


const Double1 inverse_index
{    0.,     1., 1./ 2., 1./ 3., 1./ 4., 1./ 5., 1./ 6, 1./ 7, 1./ 8., 1/ 9.,
 1./10., 1./11., 1./12., 1./13., 1./14., 1./15., 1./16, 1./17, 1./18., 1/19.,
 1./20., 1./21., 1./22., 1./23., 1./24., 1./25., 1./26, 1./27, 1./28., 1/29.,
 1./30., 1./31., 1./32., 1./33., 1./34., 1./35., 1./36, 1./37, 1./38., 1/39. };


#if (GRID_SIMD)

  // When Grid is used

  #include <Grid/Grid.h>

  typedef Grid::vRealD vReal;

  const long n_simd_lanes = vReal::Nsimd();

  typedef vector<vReal, Grid::alignedAllocator<vReal>> vReal1;

#else

  // When Grid is not used, use regular doubles

  typedef double vReal;

  const long n_simd_lanes = 1;

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




inline double getlane (
    const vReal vec,
    const int   lane  )

#if (GRID_SIMD)
{
  return vec.getlane (lane);
}
#else
{
  return vec;
}
#endif


inline double firstLane (
    const vReal vec     )
{
  return getlane (vec, 0);
}


inline double lastLane (
    const vReal vec    )
{
  return getlane (vec, n_simd_lanes-1);
}




///  vExp: exponential function for vReal types
///  !!! Only good for positive exponents !!!
///    @param[in] x: exponent
///    @return exponential of x
/////////////////////////////////////////////////

inline vReal vExp (const vReal x)

#if (GRID_SIMD)

{

  const int n = 21;

  vReal result = 1.0;

  for (int i = n; i > 1; i--)
  {
    result = vOne + x * result * inverse_index[i];
  }

  return vOne + x*result;

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

  const int n = 21;

  vReal result = 1.0;

  for (int i = n; i > 1; i--)
  {
    result = vOne + x * result * inverse_index[i];
  }

  return x*result;

}

#else

{
  return expm1 (x);
}

#endif




///  Comparator for two vReal's
///    @param[in] x : first vReal
///    @param[in] y : second vReal
///    @return true x>y in all lanes
///////////////////////////////////////////////////////////

inline bool all_greater_then (
    const vReal &x,
    const vReal &y )

#if (GRID_SIMD)

{
  GRID_FOR_ALL_LANES (lane)
  {
    if (x.getlane(lane) < y.getlane(lane)) return false;
  }

  return true;
}

#else

{
  return (x > y);
}

#endif




///  Comparator for two vReal's
///    @param[in] x : first vReal
///    @param[in] y : second vReal
///    @return true x==y in all lanes
///////////////////////////////////////////////////////////

inline bool equal (
    const vReal &x,
    const vReal &y,
    const double EPSILON )

#if (GRID_SIMD)

{
  GRID_FOR_ALL_LANES (lane)
  {
    if ( fabs(x.getlane(lane) - y.getlane(lane)) > EPSILON) return false;
  }

  return true;
}

#else

{
  return (x == y);
}

#endif




///  Comparator for two vReal's
///    @param[in] x : first vReal
///    @return true x==y in all lanes
///////////////////////////////////////////////////////////

inline double get (
    const vReal1 &x,
    const long    f )

#if (GRID_SIMD)

{
  const long index = newIndex (f);
  const  int lane  = laneNr   (f);

  return x[index].getlane(lane);
}

#else

{
  return x[f];
}

#endif




///  Pack a vector in to aligned simd-vectors
///    @param[in] vec : vector to pack
///    @return packed vector
/////////////////////////////////////////////

inline vReal1 pack (
    const Double1 &vec)

#if (GRID_SIMD)

{
  vReal1 vec_packed (reduced (vec.size ()), 0.0);

  for (long i = 0; i < vec.size(); i++)
  {
    const long index = newIndex (i);
    const  int lane  = laneNr   (i);

    vec_packed[index].putlane(vec[i], lane);
  }

  return vec_packed;
}

#else

{
  return vec;
}

#endif




///  Unpack aligned simd-vectors into an std::vector
///    @param[in] vec : vector to unpack
///    @return unpacked vector
////////////////////////////////////////////////////

inline Double1 unpack (
    const vReal1 &vec )

#if (GRID_SIMD)

{
  Double1 vec_unpacked (n_simd_lanes*vec.size (), 0.0);

  for (long i = 0; i < vec.size(); i++)
  {
    const long index = newIndex (i);
    const  int lane  = laneNr   (i);

    vec_unpacked[i] = vec[index].getlane(lane);
  }

  return vec_unpacked;
}

#else

{
  return vec;
}

#endif




inline vReal vreal(
    const double number)
{
  return vReal (number);
}



#endif // __WRAP_GRID_HPP_INCLUDED__
