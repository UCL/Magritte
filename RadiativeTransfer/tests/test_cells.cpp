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

TEST_CASE ("RAYS constructor")
{
  const int  Dimension = 1;
  const long Nrays     = 2;
  const bool Fixed_Ncells = true;
  const long Ncells    = 4;

  const CELLS<Dimension, Nrays, Fixed_Ncells, Ncells> cells(Ncells);
}
