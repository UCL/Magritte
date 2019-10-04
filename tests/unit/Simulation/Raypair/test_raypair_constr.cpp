// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include "catch.hpp"

#include "configure.hpp"
#include "Simulation/Raypair/raypair.hpp"
#include "Tools/types.hpp"
#include "Tools/Parallel/wrap_omp.hpp"
#include "Tools/logger.hpp"

#define EPS 1.0E-15


TEST_CASE ("RayPair::Raypair")
{

  const long length     = 100;
  const long n_off_diag =   0;

  int nthrds = get_nthreads ();

  cout << "number of threads = " << nthrds << endl;
  cout << omp_get_num_threads ()           << endl;

  // std::cin >> nthrds;
  nthrds = 9;

  //RayPair rayPair (length, n_off_diag);

  vector <RayPair> rayPairs (nthrds);

  for (int t = 0; t < rayPairs.size(); t++)
  {
    rayPairs[t].resize (length, n_off_diag);
  }


  //for (int i = 0; i < nthrds; i++)
  //{
  //  rayPairs.push_back (RayPair (length, n_off_diag));
  //}

  cout << "size of rayPairs = " << rayPairs.size() << endl;

  for (int i = 0; i < rayPairs.size(); i++)
  {
    cout << &rayPairs[i] << endl;
    rayPairs[i].I_bdy_0 = (vReal) 10.0;
  }

  for (int i = 0; i < rayPairs.size(); i++)
  {
    cout << rayPairs[i].I_bdy_0 << endl;
  }

}
