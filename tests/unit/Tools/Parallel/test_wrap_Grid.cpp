// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <iostream>
#include <string>
using namespace std;

#include "catch.hpp"

#include "Tools/Parallel/wrap_Grid.hpp"
#include "Tools/timer.hpp"


#define EPS 1.0E-9


TEST_CASE ("Grid-SIMD wrapper")
{


  SECTION ("OMP_PARALLEL_FOR macro")
  {
    CHECK (equal (vExp(1.0), 2.7182818284590452, EPS));


    Timer timer ("vExp");
    timer.start();


    for (int i = 0; i < 9000000; i++)
    {
      vExpMinus(vOne);
    }

    timer.stop();
    timer.print();




  }

}
