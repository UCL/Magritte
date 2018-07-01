// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <string>
#include <fstream>
#include <iostream>

#include "catch.hpp"

#include "../src/radiation.hpp"

#define EPS 1.0E-4


///  relative_error: returns relative error between A and B
///////////////////////////////////////////////////////////

double relative_error (double A, double B)
{
  return 2.0 * fabs(A-B) / fabs(A+B);
}




TEST_CASE ("Frequency interpolator")
{
	SECTION ("TEST")
	{
		long ncells = 10;
	  long nrays  = 4;
	  long nfreq  = 20;

    RADIATION radiation (ncells, nrays, nfreq, 0);
	}
  

}
