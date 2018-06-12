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
	const long ncells = 1;
	const long nspec  = 5;

	const string project_folder = "test_data/";

	const string   species_file = project_folder + "species.txt";
	const string abundance_file = project_folder + "abundance.txt";


	LINEDATA linedata;

	SPECIES species (ncells, nspec, species_file);


	for (int s = 0; s < species.nspec; s++)
	{
		cout << species.abundance[0][s] << endl;
	}


	species.density[0] = 1.0;

	// species.abundance[0] = 0.0;

	// species.abundance[4] = 1.0;


	long p = 0;
	int  l = 1;

	double temperature_gas = 10.0;


	MatrixXd C = linedata.calc_Einstein_C (species, temperature_gas, p, l);


	cout << C << endl;

	CHECK (true);

}
