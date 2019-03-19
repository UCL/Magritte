// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <iostream>
#include <string>
using namespace std;

#include "catch.hpp"

#include "configure.hpp"
#include "Model/model.hpp"
#include "Io/io_Python.hpp"


TEST_CASE ("Model", "[read]")
{

  // Setup

  const string model_folder = string (MAGRITTE_FOLDER)
                              + "/data/testdata/model_test_model.hdf5";

  IoPython io ("hdf5", model_folder);

  Model model;
  model.read (io);

  CHECK (true);

}
