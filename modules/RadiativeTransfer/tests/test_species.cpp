// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <iostream>
#include <string>
using namespace std;

#include "catch.hpp"

#include "../src/species.hpp"


#define EPS 1.0E-5


TEST_CASE ("SPECIES Constructor")
{

	long ncells = 50;

	string   species_file = "/home/frederik/Dropbox/Astro/Magritte/modules/RadiativeTransfer/tests/test_data/species.txt";
	string abundance_file = "/home/frederik/Dropbox/Astro/Magritte/modules/RadiativeTransfer/tests/test_data/abundance.txt";

	SPECIES species (ncells, 5, species_file);

	species.read (abundance_file);

	for (long p = 0; p < ncells; p++)
	{
		for (int s = 0; s < species.nspec; s++)
		{
		  cout << species.abundance[p][s] << "   ";
		}
		cout << endl;
	}


	CHECK (true);

}

