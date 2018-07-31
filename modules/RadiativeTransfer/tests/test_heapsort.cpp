// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <iostream>
#include <vector>
using namespace std;

#include "catch.hpp"
#include "tools.hpp"

#include "heapsort.hpp"

#define EPS 1.0E-7


TEST_CASE ("heapsort function")
{

  SECTION ("unsorted list")
  {
    const long length = 6;

	  vector<double> a = {3.0, 2.0, 1.0, 5.0, 4.0, 0.0};
	  vector<long>   b = {3,   2,   1,   5,   4  , 0  };

	  heapsort (a, b);

	  for (int n=0; n<length; n++)
	  {
	    CHECK (a[n] == n);
	    CHECK (a[n] == b[n]);
	  }
  }


  SECTION ("already sorted list")
  {
    const long length = 6;

	  vector<double> a = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0};
	  vector<long>   b = {0,   1,   2,   3,   4  , 5  };

	  heapsort (a, b);

	  for (int n=0; n<length; n++)
	  {
	    CHECK (a[n] == n);
	    CHECK (a[n] == b[n]);
	  }
  }

}
