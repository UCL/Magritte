// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <iostream>
using namespace std;
#include <string>

#include "catch.hpp"

#include "Io/io_Python.hpp"


TEST_CASE ("Reading input with Python")
{

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

  string name = "/home/frederik/MagritteProjects/Lines_1D_LTE/model_2019-02-18_20:21:05.hdf5";

  IoPython io2 ("io_hdf5", name);

  string datname = "Geometry/Cells/n_neighbors";

  long n;

  io2.read_length (datname, n);

  Long1 nn;

  nn.resize (n);


  io2.read_list  (datname, nn);

  for (int i = 0 ; i < nn.size(); i++)
  {
    cout << nn[i] << endl;
  }



}
