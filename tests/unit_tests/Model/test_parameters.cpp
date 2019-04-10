// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <iostream>
#include <string>
using namespace std;

#include "catch.hpp"

#include "configure.hpp"
#include "Model/parameters.hpp"
#include "Io/io_Python.hpp"


TEST_CASE ("parameters", "[io]")
{

  // Input file (in this case a folder)
  const string io_file = string (MAGRITTE_FOLDER) + "/data/testData/model_test_parameters.hdf5";

  IoPython io ("hdf5", io_file);

  Parameters parameters;

  parameters.write (io);

  // Create the input object (for txt based input)
  //IoText io (io_file);

  // Create the model object
  //Model model;
  //model.read (io);

//  SECTION ("Vector components of rays")
  //{
    //CHECK (true);
  //}

}
