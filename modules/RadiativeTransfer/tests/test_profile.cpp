// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <math.h>
#include <string>
#include <fstream>
#include <iostream>
using namespace std;
#include <Eigen/Core>
using namespace Eigen;

#include "catch.hpp"
#include "tools.hpp"

#include "profile.hpp"
#include "GridTypes.hpp"


#define EPS 1.0E-4   // Note the low accuracy !!!


TEST_CASE ("Planck function")
{
  double tmp1 = 1.0;
  vReal freq1 = 1.0;
  vReal reff1 = exp(1.0);

  vReal error1 = relative_error (Planck(tmp1, freq1), reff1);


  double tmp2 = 1.0;
  vReal freq2 = 2.3E5;
  vReal reff2 = exp(2.3E5);

  vReal error2 = relative_error (Planck(tmp2, freq2), reff2);


  double tmp3 = 1.0;
  vReal freq3 = 1.7E-2;
  vReal reff3 = exp(1.7E-2);

  vReal error3 = relative_error (Planck(tmp3, freq3), reff3);


	for (int lane = 0; lane < n_simd_lanes; lane++)
	{
//		CHECK (error1.getlane(lane) == Approx(0.0).epsilon(EPS));
//		CHECK (error2.getlane(lane) == Approx(0.0).epsilon(EPS));
//		CHECK (error3.getlane(lane) == Approx(0.0).epsilon(EPS));
	}
}




///////////////////////////

TEST_CASE ("vExp function")
{
  vReal x1    = 1.01;
  vReal reff1 = exp(1.01);

  vReal error1 = relative_error (vExp(x1), reff1);


  vReal x2    = 2.1;
  vReal reff2 = exp(2.1);

  vReal error2 = relative_error (vExp(x2), reff2);


  vReal x3    = 1.7E-4;
  vReal reff3 = exp(1.7E-4);

  vReal error3 = relative_error (vExp(x3), reff3);


	for (int lane = 0; lane < n_simd_lanes; lane++)
	{
		CHECK (error1.getlane(lane) == Approx(0.0).epsilon(EPS));
		CHECK (error2.getlane(lane) == Approx(0.0).epsilon(EPS));
		CHECK (error3.getlane(lane) == Approx(0.0).epsilon(EPS));
	}
}




////////////////////////////////

TEST_CASE ("vExpMinus function")
{
  vReal x1    = 1.01;
  vReal reff1 = exp(-1.01);

  vReal error1 = relative_error (vExpMinus(x1), reff1);


  vReal x2    = 2.1;
  vReal reff2 = exp(-2.1);

  vReal error2 = relative_error (vExpMinus(x2), reff2);


  vReal x3    = 1.7E-4;
  vReal reff3 = exp(-1.7E-4);

  vReal error3 = relative_error (vExpMinus(x3), reff3);


	for (int lane = 0; lane < n_simd_lanes; lane++)
	{
		CHECK (error1.getlane(lane) == Approx(0.0).epsilon(EPS));
		CHECK (error2.getlane(lane) == Approx(0.0).epsilon(EPS));
		CHECK (error3.getlane(lane) == Approx(0.0).epsilon(EPS));
	}
}




///////////////////////////

TEST_CASE ("vExpm1 function")
{
  vReal x1    = 1.01;
  vReal reff1 = expm1(1.01);

  vReal error1 = relative_error (vExpm1(x1), reff1);


  vReal x2    = 2.1;
  vReal reff2 = expm1(2.1);

  vReal error2 = relative_error (vExpm1(x2), reff2);


  vReal x3    = 1.7E-4;
  vReal reff3 = expm1(1.7E-4);

  vReal error3 = relative_error (vExpm1(x3), reff3);


	for (int lane = 0; lane < n_simd_lanes; lane++)
	{
		CHECK (error1.getlane(lane) == Approx(0.0).epsilon(EPS));
		CHECK (error2.getlane(lane) == Approx(0.0).epsilon(EPS));
		CHECK (error3.getlane(lane) == Approx(0.0).epsilon(EPS));
	}
}
