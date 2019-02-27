// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <iostream>
#include <string>
using namespace std;

#include "catch.hpp"

#include "Tools/wrap_mpi.hpp"


TEST_CASE ("MPI wrapper")
{

  SECTION ("MPI_for_each macro")
  {
    const long nrays = 10;

    MPI_PARALLEL_FOR (r, nrays/2)
    {
      cout << r << endl;
    }

    CHECK (true);
  }

}
