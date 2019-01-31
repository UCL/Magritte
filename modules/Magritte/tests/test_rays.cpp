// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <iostream>
#include <string>

#include "catch.hpp"

#include "rays.hpp"
#include "io_text.hpp"


#define EPS 1.0E-9   // error bar for checking doubles


TEST_CASE ("Rays")
{

  // Io file (in this case a folder)
  string io_file = "/home/frederik/Dropbox/Astro/Magritte/modules/Magritte/tests/testData/";

  // Create the io object (for txt based io)
  IoText io (io_file);

  Rays rays (io);

  for (long r = 0; r < rays.nrays; r++)
  {
    cout << rays.x[r] << endl;
  }

//
//
//  SECTION ("Vector components of rays")
//  {
//    CHECK (Approx(rays.x[0]).epsilon(EPS) == 1.0);
//    CHECK (Approx(rays.y[0]).epsilon(EPS) == 0.0);
//    CHECK (Approx(rays.z[0]).epsilon(EPS) == 0.0);
//
//    CHECK (Approx(rays.x[1]).epsilon(EPS) == -1.0);
//    CHECK (Approx(rays.y[1]).epsilon(EPS) ==  0.0);
//    CHECK (Approx(rays.z[1]).epsilon(EPS) ==  0.0);
//  }
//
//
//  SECTION ("Antipodal rays")
//  {
//    CHECK (rays.antipod[0] == 1);
//    CHECK (rays.antipod[1] == 0);
//  }
//}
//
//
//
//
///////////////////////////////////
//
//TEST_CASE ("RAYS constructor 2D")
//{
//  const int  Dimension = 2;
//  const long Nrays     = 8;
//
//  const RAYS <Dimension, Nrays> rays;
//
//
//  SECTION ("Vector components of rays")
//  {
//    CHECK (Approx(rays.x[0]).epsilon(EPS) == 1.0);
//    CHECK (Approx(rays.y[0]).epsilon(EPS) == 0.0);
//    CHECK (Approx(rays.z[0]).epsilon(EPS) == 0.0);
//
//    CHECK (Approx(rays.x[4]).epsilon(EPS) == -1.0);
//    CHECK (Approx(rays.y[4]).epsilon(EPS) ==  0.0);
//    CHECK (Approx(rays.z[4]).epsilon(EPS) ==  0.0);
//  }
//
//
//  SECTION ("Antipodal rays")
//  {
//    CHECK (rays.antipod[0] == 4);
//    CHECK (rays.antipod[4] == 0);
//
//    CHECK (rays.antipod[7] == 3);
//    CHECK (rays.antipod[3] == 7);
//  }
//}
//
//
//
//
///////////////////////////////////
//
//TEST_CASE ("RAYS constructor 3D")
//{
//  const int  Dimension = 3;
//  const long nsides    = 4;
//  const long Nrays     = 12*nsides*nsides;
//
//  const RAYS <Dimension, Nrays> rays;
//
//
//  SECTION ("Antipodal rays")
//  {
//    for (long r = 0; r < Nrays; r++)
//    {
//      long ar = rays.antipod[r];
//
//      CHECK (Approx(rays.x[r]).epsilon(EPS) == -rays.x[ar]);
//      CHECK (Approx(rays.y[r]).epsilon(EPS) == -rays.y[ar]);
//      CHECK (Approx(rays.z[r]).epsilon(EPS) == -rays.z[ar]);
//    }
//  }
//
//
//  SECTION ("xy mirror rays")
//  {
//    for (long r = 0; r < Nrays; r++)
//    {
//      long mr = rays.mirror_xz[r];
//
//      CHECK (Approx(rays.x[r]).epsilon(EPS) == +rays.x[mr]);
//      CHECK (Approx(rays.y[r]).epsilon(EPS) == -rays.y[mr]);
//      CHECK (Approx(rays.z[r]).epsilon(EPS) == +rays.z[mr]);
//    }
//  }
}
//
