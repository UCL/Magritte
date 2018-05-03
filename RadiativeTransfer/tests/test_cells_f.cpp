// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <iostream>
#include <string>

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "../src/cells_f.hpp"


#define EPS 1.0E-5

TEST_CASE ("CELLS constructor")
{
  const int  Dimension   = 1;
  const long Nrays       = 2;
  const long Ncells      = 10000;

  CELLS <Dimension, Nrays, Ncells> cells;

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
  const long Ncells      = 10000;

  CELLS <Dimension, Nrays, Ncells> cells;

  cells.initialize();

  for (long p = 0; p < Ncells; p++)
  {
    CHECK (cells.x[p] == 0.0);
    CHECK (cells.y[p] == 0.0);
    CHECK (cells.z[p] == 0.0);

    CHECK (cells.n_neighbors[p] == 0);

    CHECK (cells.vx[p] == 0.0);
    CHECK (cells.vy[p] == 0.0);
    CHECK (cells.vz[p] == 0.0);

    CHECK (cells.id[p] == p);

    CHECK (cells.removed[p]  == false);
    CHECK (cells.boundary[p] == false);
    CHECK (cells.mirror[p]   == false);
  }

}
