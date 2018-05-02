// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <iostream>
#include <string>

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "../src/cells.hpp"


#define EPS 1.0E-5

TEST_CASE ("CELLS constructor")
{
  const int  Dimension    = 1;
  const long Nrays        = 2;
  const bool FixedNcells = true;
  const long Ncells       = 4;

  CELLS <Dimension, Nrays, FixedNcells, Ncells> cells(Ncells);

  cells.initialize();

  std::cout << cells.boundary[3] << std::endl;
  std::cout << false << std::endl;
  std::cout << true  << std::endl;

}
