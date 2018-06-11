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

#include "../src/levels.hpp"


TEST_CASE ("LEVELS constructor")
{

	const long ncells = 50;

	LINEDATA linedata;

	LEVELS levels (ncells, linedata);


	cout << linedata.irad[0][0] << endl;
	cout << linedata.num[0]     << endl;

	CHECK (true);

}




TEST_CASE ("calc_Einstein_C")
{

//	SPECIES species
//	LINEDATA linedata;
//
//	linedata.calc_Einstein_C (species, temperature_gas, o, l, C);
//	CHECK (true);

}
