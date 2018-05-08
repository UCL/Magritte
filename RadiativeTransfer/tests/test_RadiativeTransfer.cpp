// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <string>
#include <fstream>
#include <iostream>
#include <Eigen/Core>
#include <Eigen/Dense>

#include "catch.hpp"

#include "../src/RadiativeTransfer.hpp"

#define EPS 1.0E-4


///  relative_error: returns relative error between A and B
///////////////////////////////////////////////////////////

double relative_error (double A, double B)
{
  return 2.0 * fabs(A-B) / fabs(A+B);
}




TEST_CASE ("Ray setup")
{
	const int Dimension = 1;
	const long Nrays    = 2;
	const long Nfreq    = 10;

	const long ncells   = 10;

	CELLS <Dimension, Nrays> Cells (ncells);
	CELLS <Dimension, Nrays> *cells = &Cells;
	
	cells->initialize();

	long freq[Nfreq];
	long rays[Nrays];

	double  source[ncells];
	double opacity[ncells];

	for (long p = 0; p < ncells; p++)
	{
     source[p] = 0.0;
	  opacity[p] = 1.0;	
	}

	RadiativeTransfer <Dimension, Nrays, Nfreq>
										(cells, freq, rays, source, opacity);

	SECTION ("1D ray")
	{
  	CHECK (true);
	}	
}
