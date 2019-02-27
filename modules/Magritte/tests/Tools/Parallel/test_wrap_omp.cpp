// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <iostream>
#include <string>
using namespace std;

#include "catch.hpp"

#include "Tools/wrap_omp.hpp"


TEST_CASE ("OMP wrapper")
{

  omp_set_dynamic     (0);   // Disable dynamic teams to enforce number of threads
  omp_set_num_threads (2);   // Set number of threads to 2


  SECTION ("OMP_PARALLEL_FOR macro")
  {
    const long ncells = 5;

    // Outside parallel region, should be single thread
    CHECK (omp_get_num_threads() == 1);


    OMP_PARALLEL_FOR (o, ncells)
    {
      // Outside parallel region, should be single thread
      CHECK (omp_get_num_threads() == 2);
      cout << omp_get_num_threads() << endl;
    }


    // Outside parallel region, should be single thread
    CHECK (omp_get_num_threads() == 1);
  }

}
