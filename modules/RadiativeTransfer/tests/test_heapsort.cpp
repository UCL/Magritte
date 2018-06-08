// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <iostream>
#include <vector>
using namespace std;

#include "catch.hpp"

#include "../src/heapsort.hpp"

#define EPS 1.0E-7


///  relative_error: returns relative error between A and B
///////////////////////////////////////////////////////////

double relative_error (double A, double B)
{
  return 2.0 * fabs(A-B) / fabs(A+B);
}


TEST_CASE ("Heapsort")
{

	vector<double> a = {3.0, 2.0, 1.0, 5.0, 4.0};
	vector<long>   b = {1, 2, 3, 4, 5};

	heapsort (a, b, 5);

	for (int n=0; n<5; n++)
	{
		cout << "a = " << a[n] << endl;
		cout << "b = " << b[n] << endl;

	  CHECK (a[n] == b[n]);
	}
	

	CHECK (true);
}
