// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <iostream>
#include <string>
using namespace std;

#include "catch.hpp"

#include "../src/set_up_ray.hpp"
#include "../src/types.hpp"


#define EPS 1.0E-5


TEST_CASE ("CELLS constructor")
{
  const int  Dimension = 1;
  const long ncells    = 50;
  const long Nrays     = 2;
  const long nspec     = 5;	

	const string       cells_file = "test_data/cells.txt";
	const string n_neighbors_file = "test_data/n_neighbors.txt";
	const string   neighbors_file = "test_data/neighbors.txt";
	const string    boundary_file = "test_data/boundary.txt";

  CELLS<Dimension, Nrays> cells (ncells, neighbors_file);
 
  cells.read (cells_file, n_neighbors_file, boundary_file);
 
  cells.boundary[0]        = true;
  cells.boundary[ncells-1] = true;
 
  cells.neighbors[0][0]        = 1;
  cells.n_neighbors[0]         = 1;
  cells.neighbors[ncells-1][0] = ncells-2;
  cells.n_neighbors[ncells-1]  = 1;
 
 
  for (long p = 1; p < ncells-1; p++)
  {
    cells.neighbors[p][0] = p-1;
    cells.neighbors[p][1] = p+1;
    cells.n_neighbors[p]  = 2;
  }


	LINEDATA linedata;

	FREQUENCIES frequencies (ncells, linedata);

  long nfreq_red = frequencies.nfreq_red;

	TEMPERATURE temperature (ncells);

	LINES lines (cells.ncells, linedata);

	SCATTERING scattering (Nrays, 1, nfreq_red);

  RADIATION radiation (ncells, Nrays, nfreq_red, 0);

  long n = 0;

  vReal2   Su (cells.ncells, vReal1 (frequencies.nfreq));    // effective source for u along ray r
  vReal2   Sv (cells.ncells, vReal1 (frequencies.nfreq));    // effective source for v along ray r
  vReal2 dtau (cells.ncells, vReal1 (frequencies.nfreq));    // optical depth increment along ray r


	long o = 0;
	long r = 0;
	long R = 0;

	set_up_ray <Dimension, Nrays>
	         	 (cells, frequencies, temperature, lines, scattering, radiation,
							o, r, R, ray, n, Su, Sv, dtau);


	cout << n << endl;

}
