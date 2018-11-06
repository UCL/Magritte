// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <iostream>
using namespace std;

#include "catch.hpp"

#include "interpolation.hpp"
#include "GridTypes.hpp"
#include "types.hpp"


#define EPS 1.0E-4


TEST_CASE ("search function")
{

  const int nElem = 30;

  Double1 x (nElem);


  // Populate x {1, 2, 3, 4, ...}

  for (int i = 0; i < nElem; i++)
  {
    x[i] = i - 0.7;
  }


  long reff1  = 7;
  long index1 = search (x, 6);

  CHECK (index1 == reff1);

  long reff2  = 0;
  long index2 = search (x, -1);

  CHECK (index2 == reff2);

  long reff3  = 29;
  long index3 = search (x, 636);

  CHECK (index3 == reff3);

  long reff4  = 1;
  long index4 = search (x, 0.3);

  CHECK (index4 == reff4);

  long reff5  = 14;
  long index5 = search (x, 14.0-0.7);

  CHECK (index5 == reff5);
}




////////////////////////////////////////

TEST_CASE ("search_with_notch function")
{

  const int nElem = 3;

  vReal1 vec (nElem);


  // Populate vec {1, 2, 3, 4, ...}

  int index = 0;

  for (int i = 0; i < nElem; i++)
  {
#   if (GRID_SIMD)
      for (int lane = 0; lane < n_simd_lanes; lane++)
      {
        vec[i].putlane(index-0.3, lane);
        index++;
      }
#   else
      vec[i] = index-0.3;
      index++;
#   endif
  }


  SECTION ("notch below value")
  {

    long notch = 3;
    long reff  = 7;

    search_with_notch (vec, notch, 6);

    CHECK (notch == reff);

  }


  SECTION ("notch above value")
  {

    long notch = 9;
    long reff  = 9;

    search_with_notch (vec, notch, 6);

    CHECK (notch == reff);

  }


  SECTION ("notch out of range")
  {

    long notch = 90;
    long reff  = n_simd_lanes*nElem-1;

    search_with_notch (vec, notch, 6);

    CHECK (notch == reff);

  }

}




//////////////////////////////////////////////////

TEST_CASE ("interpolate_linear function (double)")
{

  double x1 = 2.0;
  double x2 = 4.0;

  double f1 = 3.5;
  double f2 = 8.5;

  double reff   = 6.0;
  double result = interpolate_linear (x1, f1, x2, f2, 3.0);

  CHECK (result == reff);

}




#if (GRID_SIMD)

/////////////////////////////////////////////////

TEST_CASE ("interpolate_linear function (vReal)")
{

  vReal x1 = 2.0;
  vReal x2 = 4.0;

  vReal f1 = 3.5;
  vReal f2 = 8.5;

  vReal reff   = 6.0;
  vReal result = interpolate_linear (x1, f1, x2, f2, 3.0);

  for (int lane = 0; lane < n_simd_lanes; lane++)
  {
    CHECK (result.getlane(lane) == reff.getlane(lane));
  }

}

#endif
