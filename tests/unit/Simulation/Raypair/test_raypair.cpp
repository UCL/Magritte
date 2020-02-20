// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <Eigen/Core>
#include <Eigen/Dense>


#include "catch.hpp"

#include "configure.hpp"
#include "Simulation/Raypair/raypair.hpp"
#include "Model/Thermodynamics/thermodynamics.hpp"
#include "Tools/logger.hpp"

#define EPS 1.0E-15

// Allow access to private variables
#define private public


//double derive (double y2, double y1, double dx)

TEST_CASE ("RayPair::solve")
{

  // Setup

  const long n_r  = 5;
  const long n_ar = 5;

  const long length = n_r + n_ar;

  RayPair rayPair;
  rayPair.resize (length, 1);


  rayPair.initialize (n_ar, n_r);

  const long ndep       = rayPair.ndep;
  const long n_off_diag = rayPair.n_off_diag;

  const vReal  eta      = 1.0;
  const vReal  chi      = 0.3;
  const vReal  U_scaled = 0.0;
  const vReal  V_scaled = 0.0;
  const double dZ       = 0.1;

  rayPair.I_bdy_0 = 0.0;
  rayPair.I_bdy_n = 0.0;

  for (long d = 0; d < ndep; d++)
  {
    rayPair.set_term1_and_term2 ((d+1)*eta, (d*d)*chi,       U_scaled, V_scaled, d);
    rayPair.set_dtau            ((d*d)*chi, (d-1)*(d-1)*chi, dZ,                 d);
  }


  rayPair.solve ();


   SECTION ("u and v consistency")
   {

    for (long d = 0; d < ndep-1; d++)
    {

      cout << rayPair.dtau[d] << endl;

      // Why are they NaN ???
      cout << rayPair.Su[d] << "\t" << rayPair.Sv[d] << endl;

      //cout << rayPair.Sv[d] << rayPair.Sv[d] + (rayPair.Su[d+1] - rayPair.Su[d]) / rayPair.dtau[d] << endl;
    }

   }


  SECTION ("Lambda operator")
  {
    Eigen::MatrixXd M (ndep, ndep);

#   if (GRID_SIMD)
      for (long d = 0; d < ndep-1; d++)
      {
        M(d,d+1) = -rayPair.C[d].getlane(0);
      }

      for (long d = 1; d < ndep; d++)
      {
        M(d,d-1) = -rayPair.A[d].getlane(0);
      }

      for (long d = 1; d < ndep-1; d++)
      {
        M(d,d) = 1.0 + rayPair.A[d].getlane(0) + rayPair.C[d].getlane(0);
      }

      M(0     ,0      ) = 1 + 2.0/rayPair.dtau[0].getlane(0)
                            + 2.0/(rayPair.dtau[0].getlane(0)     *rayPair.dtau[0].getlane(0)     );
      M(ndep-1, ndep-1) = 1 + 2.0/rayPair.dtau[ndep-2].getlane(0)
                            + 2.0/(rayPair.dtau[ndep-2].getlane(0)*rayPair.dtau[ndep-2].getlane(0));
#   else
      for (long d = 0; d < ndep-1; d++)
      {
        M(d,d+1) = -rayPair.C[d];
      }

      for (long d = 1; d < ndep; d++)
      {
        M(d,d-1) = -rayPair.A[d];
      }

      for (long d = 1; d < ndep-1; d++)
      {
        M(d,d) = 1.0 + rayPair.A[d] + rayPair.C[d];
      }

      M(0     ,0      ) = 1 + 2.0 /  rayPair.dtau[0]
                            + 2.0 / (rayPair.dtau[0]      * rayPair.dtau[0]     );
      M(ndep-1, ndep-1) = 1 + 2.0 /  rayPair.dtau[ndep-2]
                            + 2.0 / (rayPair.dtau[ndep-2] * rayPair.dtau[ndep-2]);
#   endif


    MatrixXd M_inverse = M.inverse();


    SECTION ("L diagonal")
    {
      for (long d = 0; d < ndep; d++)
      {
        CHECK (equal (rayPair.L_diag[d], M_inverse(d,d), EPS));
      }
    }

    SECTION ("L upper")
    {
      for (long m = 0; m < n_off_diag; m++)
      {
        for (long d = 0; d < ndep-m-1; d++)
        {
          cout << "m = " << m << "   d = " << d << endl;

          CHECK (equal (rayPair.L_upper[m][d], M_inverse(d,d+m+1), EPS));
        }
      }
    }

    SECTION ("L lower")
    {
      for (long m = 0; m < n_off_diag; m++)
      {
        for (long d = 0; d < ndep-m-1; d++)
        {
          cout << "m = " << m << "   d = " << d << endl;

          CHECK (equal (rayPair.L_lower[m][d], M_inverse(d+m+1,d), EPS));
        }
      }
    }
  }

}




TEST_CASE ("RayPair::get_L_diag")
{

  // Setup

  const long n_r  = 5;
  const long n_ar = 5;

  const long length = n_r + n_ar;

  RayPair rayPair;
  rayPair.resize (length, 1);

  rayPair.initialize (n_ar, n_r);

  const long ndep       = rayPair.ndep;
  const long n_off_diag = rayPair.n_off_diag;

  const vReal  eta      = 1.0E+0;
  const vReal  chi      = 1.0E+9;
  const vReal  U_scaled = 0.0;
  const vReal  V_scaled = 0.0;
  const double dZ       = 1.0E+0;

  for (long d = 0; d < ndep; d++)
  {
    rayPair.set_term1_and_term2 ((d+1)*eta, (d*d)*chi,       U_scaled, V_scaled, d);
    rayPair.set_dtau            ((d*d)*chi, (d-1)*(d-1)*chi, dZ,                 d);
  }


  rayPair.solve ();

  SECTION ("Lambda operator")
  {
    Eigen::MatrixXd M (ndep, ndep);

#   if (GRID_SIMD)
      for (long d = 0; d < ndep-1; d++)
      {
        M(d,d+1) = -rayPair.C[d].getlane(0);
      }

      for (long d = 1; d < ndep; d++)
      {
        M(d,d-1) = -rayPair.A[d].getlane(0);
      }

      for (long d = 1; d < ndep-1; d++)
      {
        M(d,d) = 1.0 + rayPair.A[d].getlane(0) + rayPair.C[d].getlane(0);
      }

      M(0     ,0      ) = 1 + 2.0/rayPair.dtau[0].getlane(0)
                            + 2.0/(rayPair.dtau[0].getlane(0)     *rayPair.dtau[0].getlane(0)     );
      M(ndep-1, ndep-1) = 1 + 2.0/rayPair.dtau[ndep-2].getlane(0)
                            + 2.0/(rayPair.dtau[ndep-2].getlane(0)*rayPair.dtau[ndep-2].getlane(0));
#   else
      for (long d = 0; d < ndep-1; d++)
      {
        M(d,d+1) = -rayPair.C[d];
      }

      for (long d = 1; d < ndep; d++)
      {
        M(d,d-1) = -rayPair.A[d];
      }

      for (long d = 1; d < ndep-1; d++)
      {
        M(d,d) = 1.0 + rayPair.A[d] + rayPair.C[d];
      }

      M(0     ,0      ) = 1 + 2.0/rayPair.dtau[0]
                            + 2.0/(rayPair.dtau[0]     *rayPair.dtau[0]     );
      M(ndep-1, ndep-1) = 1 + 2.0/rayPair.dtau[ndep-2]
                            + 2.0/(rayPair.dtau[ndep-2]*rayPair.dtau[ndep-2]);
#   endif



    MatrixXd M_inverse = M.inverse();


    //cout << M_inverse << endl;


    SECTION ("L diagonal")
    {
      for (long d = 0; d < ndep; d++)
      {
        CHECK (equal (rayPair.L_diag[d], M_inverse(d,d), EPS));
      }
    }

    SECTION ("L upper")
    {
      for (long m = 0; m < n_off_diag; m++)
      {
        for (long d = 0; d < ndep-m-1; d++)
        {
          cout << "m = " << m << "   d = " << d << endl;

          CHECK (equal (rayPair.L_upper[m][d], M_inverse(d,d+m+1), EPS));
        }
      }
    }

    SECTION ("L lower")
    {
      for (long m = 0; m < n_off_diag; m++)
      {
        for (long d = 0; d < ndep-m-1; d++)
        {
          cout << "m = " << m << "   d = " << d << endl;

          CHECK (equal (rayPair.L_lower[m][d], M_inverse(d+m+1,d), EPS));
        }
      }
    }
  }

}
