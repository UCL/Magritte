// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <iostream>
#include <string>
using namespace std;

#include "catch.hpp"

#include "configure.hpp"
#include "Io/python/io_python.hpp"
#include "Tools/types.hpp"


TEST_CASE ("IoPython hdf5")
{

  // Setup

  const string model_folder = string (MAGRITTE_FOLDER)
                              + "bin/tests/testdata/test_io_python.hdf5";

  IoPython io ("hdf5", model_folder);

  string name = model_folder + "Lines/LineProducingSpecies_0/population";

  Double2 pop (1, Double1 (1));

  cout << &pop << endl;

  int err = io.read_array (name, pop);

  cout << err << endl;


  //cout << endl;
  //cout << endl;
  //cout << endl;
  //cout << endl;
  //cout << endl;


  //cout << pop[0][0] << endl;
  //string input = "/home/frederik/Desktop/Magritte/modules/Magritte/tests/testData/test1.hdf5";

  //IoPython io ("io_hdf5", input);
  ////IoPython io ("/home/frederik/Desktop/Magritte/modules/Magritte/setup/");


  //SECTION ("Vector components of rays")
  //{
  //  cout << "Fine" << endl;

  //  //long ntest;
  //  //io.read_length ("test", ntest);

  //  //cout << "Stil fine" << endl;
  //  //cout << nboundary << endl;

  //  Double1 test1 (2);
  //  //Double1 test2 (3);

  //  io.read_list("data", test1);

  //  for (long n = 0; n < test1.size(); n++)
  //  {
  //    cout << test1[n] << endl;
  //  }


  //  //io.read_list("test", test2);

  //  //for (long n = 0; n < test2.size(); n++)
  //  //{
  //  //  cout << test2[n] << endl;
  //  //}

  //  CHECK (true);
  //}

  //string name = "/home/frederik/MagritteProjects/Lines_1D_LTE/model_2019-02-//18_20:21:05.hdf5";

  //IoPython io2 ("io_hdf5", name);

  //string datname = "Geometry/Cells/n_neighbors";

  //long n;

  //io2.read_length (datname, n);

  //Long1 nn;

  //nn.resize (n);


  //io2.read_list  (datname, nn);

  //for (int i = 0 ; i < nn.size(); i++)
  //{
  //  cout << nn[i] << endl;
  //}



}
