// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <iostream>
#include <string>

#include "catch.hpp"

#include "Model/model.hpp"
#include "Io/io_text.hpp"


TEST_CASE ("Text input")
{

  // Input file (in this case a folder)
  string io_file = "/home/frederik/Dropbox/Astro/Magritte/modules/Magritte/tests/testData/";

  // Create the input object (for txt based input)
  IoText io (io_file);

  // Create the model object
  Model model;
  model.read (io);

//  SECTION ("Vector components of rays")
  //{
    //CHECK (true);
  //}

}
