// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <string>
#include <fstream>
#include <iostream>

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

	double numbers[] = {1.0, 3.4, 4.2, 5.4, 9.3, 13.4};

  long start = 1;
	long stop  = 4;

	CHECK (search (numbers, start, stop, 2.9) == 1);
	CHECK (search (numbers, start, stop, 4.1) == 2);
	CHECK (search (numbers, start, stop, 5.1) == 3);

}




TEST_CASE ("Simple interpolation")
{

	double numbers[] = {2.0, 4.0, 6.0, 8.0, 10.0, 12.0};
	double  values[] = {1.0, 2.0, 3.0, 4.0,  5.0,  6.0};

  long start = 1;
	long stop  = 4;

	CHECK (interpolate (values, numbers, start, stop,  0.9) == 2.0);
	CHECK (interpolate (values, numbers, start, stop,  5.0) == 2.5);
	CHECK (interpolate (values, numbers, start, stop,  6.0) == 3.0);
	CHECK (interpolate (values, numbers, start, stop, 11.1) == 4.0);

}
