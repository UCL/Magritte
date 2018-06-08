// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <string>
#include <vector>
#include <iostream>
using namespace std;
#include <Eigen/Core>
using namespace Eigen;


#include "catch.hpp"

#include "../src/linedata.hpp"


TEST_CASE ("Constructor")
{

	LINEDATA linedata;

	cout << linedata.irad[0][0] << endl;
	cout << linedata.num[0] << endl;

	CHECK (true);

}




TEST_CASE ("calc_Einstein_C")
{

	SPECIES species
	LINEDATA linedata;

	linedata.calc_Einstein_C (species, temperature_gas, o, l, C);
	CHECK (true);

}
