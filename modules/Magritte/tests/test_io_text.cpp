// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <iostream>
#include <string>

#include "catch.hpp"

#include "io_text.hpp"


TEST_CASE ("Reading .txt input")
{

  IoText io ("testData/cells.txt");


  SECTION ("Vector components of rays")
  {
    CHECK (true);
  }

}
