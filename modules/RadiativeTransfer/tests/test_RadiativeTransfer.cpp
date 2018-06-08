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
#include "../src/cells.hpp"
#include "../src/medium.hpp"
#include "../src/radiation.hpp"

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

	long nfreq_l = 1;
	long nfreq_c = 1;
	long nfreq_s = 1;

	CELLS <Dimension, Nrays> cells (ncells);
	
	cells.initialize ();


 
	for (long p = 0; p < ncells; p++)
	{
		cells.x[p] = 1.23 * p;
	}

  cells.boundary[0]        = true;
  cells.boundary[ncells-1] = true;

	cells.neighbor[RINDEX(0,0)]        = 1;
	cells.neighbor[RINDEX(ncells-1,0)] = ncells-2;


	for (long p = 1; p < ncells-1; p++)
	{
		cells.neighbor[RINDEX(p,0)] = p-1;
		cells.neighbor[RINDEX(p,1)] = p+1;
	}

	LINES lines;

	SCATTERING scattering;

  RADIATION radiation (ncells, Nrays, Nfreq);


	long freq[Nfreq];
	long rays[Nrays] = {0, 1};



	RadiativeTransfer <Dimension, Nrays, Nfreq>
										(cells, Nrays, rays, lines, scattering, radiation);


  std::cout << "I'm fine" << std::endl;  

}
