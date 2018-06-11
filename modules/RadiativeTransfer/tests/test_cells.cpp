// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <iostream>
#include <string>

#include "catch.hpp"

#include "../src/cells.hpp"


#define EPS 1.0E-5


TEST_CASE ("CELLS constructor")
{
  const int  Dimension   = 1;
  const long Nrays       = 2;
  const long Ncells      = 50;

	string n_neighbors_file = "test_data/n_neighbors.txt";

  CELLS <Dimension, Nrays> cells (Ncells, n_neighbors_file);


  SECTION ("RAYS in CELLS")
  {
    CHECK (Approx(cells.rays.x[0]).epsilon(EPS) == 1.0);
    CHECK (Approx(cells.rays.y[0]).epsilon(EPS) == 0.0);
    CHECK (Approx(cells.rays.z[0]).epsilon(EPS) == 0.0);

    CHECK (Approx(cells.rays.x[1]).epsilon(EPS) == -1.0);
    CHECK (Approx(cells.rays.y[1]).epsilon(EPS) ==  0.0);
    CHECK (Approx(cells.rays.z[1]).epsilon(EPS) ==  0.0);
  }


  SECTION ("Antipodal rays in CELLS")
  {
    CHECK (cells.rays.antipod[0] == 1);
    CHECK (cells.rays.antipod[1] == 0);
  }


  SECTION ("Check memory availability")
  {
    for (long p = 0; p < Ncells; p++)
    {
      cells.x[p] = 1.23 * p + 0.45;
    }

    for (long p = 0; p < Ncells; p++)
    {
      CHECK (cells.x[p] == 1.23 * p + 0.45);
    }
  }
}



TEST_CASE ("CELLS initialize")
{
  const int  Dimension   = 1;
  const long Nrays       = 2;
  const long Ncells      = 50;

	string n_neighbors_file = "test_data/n_neighbors.txt";

  CELLS <Dimension, Nrays> cells (Ncells, n_neighbors_file);


  for (long p = 0; p < Ncells; p++)
  {
    CHECK (cells.x[p] == 0.0);
    CHECK (cells.y[p] == 0.0);
    CHECK (cells.z[p] == 0.0);

    CHECK (cells.vx[p] == 0.0);
    CHECK (cells.vy[p] == 0.0);
    CHECK (cells.vz[p] == 0.0);

    CHECK (cells.id[p] == p);

    CHECK (cells.removed[p]  == false);
    CHECK (cells.boundary[p] == false);
    CHECK (cells.mirror[p]   == false);
  }
	

  CHECK (cells.n_neighbors[0]        == 1);
  CHECK (cells.n_neighbors[Ncells-1] == 1);

  for (long p = 1; p < Ncells-1; p++)
  {
    CHECK (cells.n_neighbors[p] == 2);
  }
}




TEST_CASE ("Read")
{
  const int  Dimension   = 1;
  const long Nrays       = 2;
  const long Ncells      = 50;

	string       cells_file = "test_data/cells.txt";
	string n_neighbors_file = "test_data/n_neighbors.txt";
	string   neighbors_file = "test_data/neighbors.txt";
	string    boundary_file = "test_data/boundary.txt";

  CELLS <Dimension, Nrays> cells (Ncells, n_neighbors_file);
	
  cells.read (cells_file, neighbors_file, boundary_file);

  for (long p = 0; p < Ncells; p++)
  {
    cout << cells.x[p] << endl;
  }

  CHECK (true);   
}




//TEST_CASE ("Ray tracer: next function 1D")
//{
//  const int  Dimension   = 1;
//  const long Nrays       = 2;
//  const long Ncells      = 10000;
//
//  CELLS <Dimension, Nrays> cells (Ncells);
//
//  for (long p = 0; p < Ncells; p++)
//  {
//    cells.x[p] = 1.0E10 * (p + 0.012345);
//  }
//}

