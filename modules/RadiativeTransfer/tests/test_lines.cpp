// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <string>
#include <fstream>
#include <iostream>
using namespace std;
#include <Eigen/Core>
using namespace Eigen;

#include "catch.hpp"

#include "../src/lines.hpp"
#include "Lines/src/linedata.hpp"

#define EPS 1.0E-7


///  relative_error: returns relative error between A and B
///////////////////////////////////////////////////////////

double relative_error (double A, double B)
{
  return 2.0 * fabs(A-B) / fabs(A+B);
}




TEST_CASE ("Constructor")
{
	long ncells = 1;

	LINEDATA linedata;

	LINES lines (ncells, linedata);

	for (int l = 0; l < linedata.nlspec; l++)
	{
		cout << lines.nrad_cum[l] << endl;
	}
	
	CHECK (true);
}




TEST_CASE ("get_emissivity_and_opacity")
{

}




TEST_CASE ("add_emissivity_and_opacity")
{

}
