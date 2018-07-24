// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <iostream>
#include <vector>
using namespace std;

#include "catch.hpp"

#include "../src/interpolation.hpp"

#define EPS 1.0E-4


///  relative_error: returns relative error between A and B
///////////////////////////////////////////////////////////

double relative_error (double A, double B)
{
  return 2.0 * fabs(A-B) / fabs(A+B);
}




TEST_CASE ("Simple search")
{

	vector<double> x = {1.0, 3.4, 4.2, 5.4, 9.3, 13.4};

  long start = 1;
	long stop  = 4;

	CHECK (search (x, start, stop, 2.9) == 1);
	CHECK (search (x, start, stop, 4.1) == 2);
	CHECK (search (x, start, stop, 5.1) == 3);

}




TEST_CASE ("Simple interpolation")
{

	vector<double> x = {2.0, 4.0, 6.0, 8.0, 10.0, 12.0};
	vector<double> f = {1.0, 2.0, 3.0, 4.0,  5.0,  6.0};

  long start = 1;
	long stop  = 4;

	CHECK (interpolate (f, x, start, stop,  0.9) == 2.0);
	CHECK (interpolate (f, x, start, stop,  5.0) == 2.5);
	CHECK (interpolate (f, x, start, stop,  6.0) == 3.0);
	CHECK (interpolate (f, x, start, stop, 11.1) == 4.0);

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
