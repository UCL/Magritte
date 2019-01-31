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

  IoPython io ("/home/frederik/Desktop/Magritte/modules/Magritte/tests/testData/");


  SECTION ("Vector components of rays")
  {
    cout << "Fine" << endl;

    long nboundary = io.get_length ("boundary");

    cout << "Stil fine" << endl;
    cout << nboundary << endl;

    Long1 boundary (nboundary);

    io.read_list("boundary", boundary);

    for (long n = 0; n < boundary.size(); n++)
    {
      cout << boundary[n] << endl;
    }

    CHECK (true);
  }

}
