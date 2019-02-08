// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <iostream>
#include <string>

#include "catch.hpp"

#include "io_Python.hpp"


TEST_CASE ("Reading input with Python")
{

  string input = "/home/frederik/Desktop/Magritte/modules/Magritte/tests/testData/model.hdf5";

  IoPython io ("io_hdf5", input);
  //IoPython io ("/home/frederik/Desktop/Magritte/modules/Magritte/setup/");


  SECTION ("Vector components of rays")
  {
    cout << "Fine" << endl;

    //long ntest;
    //io.read_length ("test", ntest);

    //cout << "Stil fine" << endl;
    //cout << nboundary << endl;

    Double1 test1 (3);
    Double1 test2 (3);

    io.read_list("test", test1);

    for (long n = 0; n < test1.size(); n++)
    {
      cout << test1[n] << endl;
    }


    io.read_list("test", test2);

    for (long n = 0; n < test2.size(); n++)
    {
      cout << test2[n] << endl;
    }

    CHECK (true);
  }

}
