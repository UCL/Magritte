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


//TEST_CASE ("Simple search")
//{
//
//	vector<double> x = {1.0, 3.4, 4.2, 5.4, 9.3, 13.4};
//
//  long start = 1;
//	long stop  = 4;
//
//	CHECK (search (x, start, stop, 2.9) == 1);
//	CHECK (search (x, start, stop, 4.1) == 2);
//	CHECK (search (x, start, stop, 5.1) == 3);
//
//}
//
//
//
//
//TEST_CASE ("Simple interpolation")
//{
//
//	vector<double> x = {2.0, 4.0, 6.0, 8.0, 10.0, 12.0};
//	vector<double> f = {1.0, 2.0, 3.0, 4.0,  5.0,  6.0};
//
//  long start = 1;
//	long stop  = 4;
//
//	CHECK (interpolate (f, x, start, stop,  0.9) == 2.0);
//	CHECK (interpolate (f, x, start, stop,  5.0) == 2.5);
//	CHECK (interpolate (f, x, start, stop,  6.0) == 3.0);
//	CHECK (interpolate (f, x, start, stop, 11.1) == 4.0);
//
//}

TEST_CASE ("search_with_notch function (double)")
{

  vReal1 vec = {1.0, 2.0, 3.0, 4.0, 5.0};

  double notch = 3;

  double reff   = 6.0;
  double result = search_with_notch (vec, notch, value);

  CHECK (result == reff);

}




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

//TEST_CASE ("Simple resample")
//{
//	vector<double> x = {2.0, 4.0, 6.0, 8.0, 10.0, 12.0};
//	vector<double> f = {1.0, 2.0, 3.0, 4.0,  5.0,  6.0};
//
//	double scale = 0.5;
//
//  long start = 1;
//	long stop  = 5;
//
//	vector<double> x_new = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
//	vector<double> f_new = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
//
//  resample (x, f, start, stop, x_new, f_new);
//
//	CHECK (f_new[0] == 0.0);
//	CHECK (f_new[1] == 2.0);
//	CHECK (f_new[2] == 2.0);
//	CHECK (f_new[3] == 2.0);
//	CHECK (f_new[4] == 2.5);
//	CHECK (f_new[5] == 0.0);
//}
