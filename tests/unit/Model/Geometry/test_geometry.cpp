// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <iostream>
#include <string>

#include "catch.hpp"

#include "Model/Geometry/geometry.hpp"


#define EPS 1.0E-5


TEST_CASE ("Geometry functions")
{

  SECTION ("get_doppler_shift")
  {

    // Set up a geometry with two points
    // (We only need their rays and velocities.)

    Geometry geometry;

    geometry.rays.x = {{1.0}, {1.0}, {1.0}, {1.0}};
    geometry.rays.y = {{0.0}, {0.0}, {0.0}, {0.0}};
    geometry.rays.z = {{0.0}, {0.0}, {0.0}, {0.0}};


    geometry.cells.vx = {0.2, 0.3, 0.1, -0.1};   // (fractions of speed of light)
    geometry.cells.vy = {0.0, 0.0, 0.0,  0.0};   // (fractions of speed of light)
    geometry.cells.vz = {0.0, 0.0, 0.0,  0.0};   // (fractions of speed of light)

    const long origin  = 0;
    const long ray     = 0;


    SECTION ("Co-moving frame")
    {
      const Frame frame = CoMoving;

      CHECK (geometry.get_doppler_shift <frame> (origin, ray, 0) == 1.0);
      CHECK (geometry.get_doppler_shift <frame> (origin, ray, 1) == 0.9);
      CHECK (geometry.get_doppler_shift <frame> (origin, ray, 2) == 1.1);
      CHECK (geometry.get_doppler_shift <frame> (origin, ray, 3) == 1.3);
    }

    SECTION ("Rest frame")
    {
      const Frame frame = Rest;

      CHECK (geometry.get_doppler_shift <frame> (origin, ray, 0) == 0.8);
      CHECK (geometry.get_doppler_shift <frame> (origin, ray, 1) == 0.7);
      CHECK (geometry.get_doppler_shift <frame> (origin, ray, 2) == 0.9);
      CHECK (geometry.get_doppler_shift <frame> (origin, ray, 3) == 1.1);
    }

  }


}


//  const int  Dimension   = 1;
//  const long Nrays       = 2;
//  const long Ncells      = 50;
//
//	string n_neighbors_file = "test_data/n_neighbors.txt";
//
//  CELLS <Dimension, Nrays> cells (Ncells, n_neighbors_file);
//
//
//  SECTION ("RAYS in CELLS")
//  {
//    CHECK (Approx(cells.rays.x[0]).epsilon(EPS) == 1.0);
//    CHECK (Approx(cells.rays.y[0]).epsilon(EPS) == 0.0);
//    CHECK (Approx(cells.rays.z[0]).epsilon(EPS) == 0.0);
//
//    CHECK (Approx(cells.rays.x[1]).epsilon(EPS) == -1.0);
//    CHECK (Approx(cells.rays.y[1]).epsilon(EPS) ==  0.0);
//    CHECK (Approx(cells.rays.z[1]).epsilon(EPS) ==  0.0);
//  }
//
//
//  SECTION ("Antipodal rays in CELLS")
//  {
//    CHECK (cells.rays.antipod[0] == 1);
//    CHECK (cells.rays.antipod[1] == 0);
//  }
//
//
//  SECTION ("Check memory availability")
//  {
//    for (long p = 0; p < Ncells; p++)
//    {
//      cells.x[p] = 1.23 * p + 0.45;
//    }
//
//    for (long p = 0; p < Ncells; p++)
//    {
//      CHECK (cells.x[p] == 1.23 * p + 0.45);
//    }
//  }
//}
//
//
//
//TEST_CASE ("CELLS initialize")
//{
//  const int  Dimension   = 1;
//  const long Nrays       = 2;
//  const long Ncells      = 50;
//
//	string n_neighbors_file = "test_data/n_neighbors.txt";
//
//  CELLS <Dimension, Nrays> cells (Ncells, n_neighbors_file);
//
//
//  for (long p = 0; p < Ncells; p++)
//  {
//    CHECK (cells.x[p] == 0.0);
//    CHECK (cells.y[p] == 0.0);
//    CHECK (cells.z[p] == 0.0);
//
//    CHECK (cells.vx[p] == 0.0);
//    CHECK (cells.vy[p] == 0.0);
//    CHECK (cells.vz[p] == 0.0);
//
//    CHECK (cells.id[p] == p);
//
//    CHECK (cells.removed[p]  == false);
//    CHECK (cells.boundary[p] == false);
//    CHECK (cells.mirror[p]   == false);
//  }
//
//
//  CHECK (cells.n_neighbors[0]        == 1);
//  CHECK (cells.n_neighbors[Ncells-1] == 1);
//
//  for (long p = 1; p < Ncells-1; p++)
//  {
//    CHECK (cells.n_neighbors[p] == 2);
//  }
//}
//
//
//
//
//TEST_CASE ("Read")
//{
//  const int  Dimension   = 1;
//  const long Nrays       = 2;
//  const long Ncells      = 50;
//
//	string       cells_file = "test_data/cells.txt";
//	string n_neighbors_file = "test_data/n_neighbors.txt";
//	string   neighbors_file = "test_data/neighbors.txt";
//	string    boundary_file = "test_data/boundary.txt";
//
//  CELLS <Dimension, Nrays> cells (Ncells, n_neighbors_file);
//
//  cells.read (cells_file, neighbors_file, boundary_file);
//
//  for (long p = 0; p < Ncells; p++)
//  {
//    cout << cells.x[p] << endl;
//  }
//
//  CHECK (true);
//}
//
//
//
//
//TEST_CASE ("Ray tracer: next function 1D")
//{
//  const int  Dimension   = 3;
//  const long Nrays       = 12;
//  const long Ncells      = 64;
//
//	const string project_folder = "test_data/Cube_64/";
//
//	const string       cells_file = project_folder + "cells.txt";
//	const string n_neighbors_file = project_folder + //"n_neighbors.txt";
//	const string   neighbors_file = project_folder + "neighbors.txt";
//	const string    boundary_file = project_folder + "boundary.txt";
//
//  CELLS <Dimension, Nrays> cells (Ncells, n_neighbors_file);
//
//  cells.read (cells_file, neighbors_file, boundary_file);
//
//
//
//
//
//  for (long p = 0; p < Ncells; p++)
//  {
//    cout << cells.x[p] << endl;
//  }
//
//  CHECK (true);
//}
//
