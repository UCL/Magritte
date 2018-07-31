// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <string>
#include <fstream>
#include <iostream>

#include "catch.hpp"
#include "tools.hpp"

#include "radiation.hpp"

#define EPS 1.0E-4


TEST_CASE ("Frequency interpolator")
{
	SECTION ("TEST")
	{
		long ncells    = 10;
	  long nrays     = 4;
	  long nfreq     = 20;
		long nboundary = 2;

    RADIATION radiation (ncells, nrays, nrays, nfreq, nboundary, 0);
	}


}


TEST_CASE ("rescale_U_and_V function")
{

	const long ncells    = 10;
  const long nrays     = 4;
  const long nfreq     = 20;
	const long nboundary = 2;

  RADIATION radiation (ncells, nrays, nrays, nfreq, nboundary, 0);



}
