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
#include "../src/lines.hpp"
#include "../src/temperature.hpp"
#include "../src/frequencies.hpp"
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
  const long nspec    = 5;
	const long ncells   = 50;

  string     species_file = "/home/frederik/Dropbox/Astro/Magritte/modules/RadiativeTransfer/tests/test_data/species.txt";
  string   abundance_file = "/home/frederik/Dropbox/Astro/Magritte/modules/RadiativeTransfer/tests/test_data/abundance.txt";
	string n_neighbors_file = "test_data/n_neighbors.txt";


	CELLS <Dimension, Nrays> cells (ncells, n_neighbors_file);
	
 
	for (long p = 0; p < ncells; p++)
	{
		cells.x[p] = 1.23 * p;
	}

  cells.boundary[0]        = true;
  cells.boundary[ncells-1] = true;

	cells.neighbors[0][0]        = 1;
  cells.n_neighbors[0]        = 1;
	cells.neighbors[ncells-1][0] = ncells-2;
  cells.n_neighbors[ncells-1] = 1;


	for (long p = 1; p < ncells-1; p++)
	{
		cells.neighbors[p][0] = p-1;
		cells.neighbors[p][1] = p+1;
    cells.n_neighbors[p] = 2;
	}


  SPECIES species (ncells, nspec, species_file);
 
  species.read (abundance_file);


	LINEDATA linedata;

	TEMPERATURE temperature (ncells);

	for (long p = 0; p < ncells; p++)
	{
    temperature.gas[p] = 100.0;
	}


	LEVELS levels (ncells, linedata);

	levels.set_LTE_populations (linedata, species, temperature);


	LINES lines (cells.ncells, linedata);

	for (long p = 0; p < ncells; p++)
	{
		for (int l = 0; l < linedata.nlspec; l++)
		{
	    lines.get_emissivity_and_opacity (linedata, levels, p, l);
		}
	}


	FREQUENCIES frequencies (ncells, linedata);

	frequencies.reset (linedata, temperature);

  long nfreq = frequencies.nfreq;

	SCATTERING scattering (Nrays, 1);

  RADIATION radiation (ncells, Nrays, nfreq);

	vector<vector<double>> J (cells.ncells, vector<double> (frequencies.nfreq));

	long rays[Nrays] = {0, 1};


  RadiativeTransfer<Dimension, Nrays>
	                 (cells, temperature, frequencies, Nrays, rays, lines, scattering, radiation, J);


	for (long p = 0; p < ncells; p++)
	{
		cout << J[p][0] << endl;
	}


}
