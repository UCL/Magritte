// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <fstream>
#include <iostream>
using namespace std;
#include <Eigen/Core>
using namespace Eigen;

#include "catch.hpp"
#include "tools.hpp"

#include "raypair.hpp"
#include "solve_ray.hpp"
#include "GridTypes.hpp"
#include "folders.hpp"

#define EPS 1.0E-5


///  feautrier_error: returns difference between the two sides of the Feautrier eq.
///    @param[in] S: pointer to source function
///    @param[in] dtau: pointer to optical depth increments
///    @param[in] u: pointer to source function
///    @param[in] i: index of point where Feautrier equation is to be evaluated
///////////////////////////////////////////////////////////////////////////////////

vReal feautrier_error (const vReal* S, const vReal* dtau, const vReal1 u, const long i)
{

  // Left hand side of Feautrier equation (d^2u / dtau^2 - u)

  vReal lhs =  ( (u[i+1] - u[i]) / dtau[i+1] - (u[i] - u[i-1]) / dtau[i] )
               / (0.5 * (dtau[i+1] + dtau[i])) - u[i];


  // Right hand side of Feautrier equation (-S)

  vReal rhs = -S[i];

  return relative_error (rhs, lhs);

}




////////////////////////////////////////////////

TEST_CASE ("Feautrier solver on feautrier1.txt")
{

  const long ndep  = 100;
  const long ndiag = ndep;

  vReal2 U;
  vReal2 V;
  vReal3 Ibdy;
  Long1 c2b;

  RAYPAIR raypair (ndep, 1, 0, 1, 0, U, V, Ibdy, c2b);

  vReal      S[ndep];
  vReal   dtau[ndep];
  vReal      u[ndep];
  vReal      v[ndep];
  vReal Lambda[ndep];

  double S_local, dtau_local, u_sol;


  // Reading test data

  ifstream infile (Magritte_folder + "tests/test_data/feautrier1.txt");

  for (long i = 0; i < ndep; i++)
  {
    infile >> S_local >> dtau_local >> u_sol;

    raypair.Su[i] = raypair.Sv[i] = S_local;
                  raypair.dtau[i] = dtau_local;

    u[i] = v[i] = S[i] =    S_local;
               dtau[i] = dtau_local;
  }

  raypair.ndep = ndep;

  raypair.solve ();

  solve_ray (ndep, u, v, dtau, ndiag, Lambda, ndep);


  SECTION ("Feautrier equation")
  {

    // Check if result satisfies the Feautrier equation (d^2u/dtau^2=u-S)

    for (long m = 1; m < ndep-1; m++)
    {
      vReal error_u = feautrier_error (S, dtau, raypair.Su, m);
      vReal error_v = feautrier_error (S, dtau, raypair.Sv, m);


#     if (GRID_SIMD)
        for (int lane = 0; lane < n_simd_lanes; lane++)
        {
          CHECK (error_u.getlane(lane) == Approx(0.0).epsilon(EPS));
          CHECK (error_v.getlane(lane) == Approx(0.0).epsilon(EPS));
        }
#     else
        CHECK (error_u == Approx(0.0).epsilon(EPS));
        CHECK (error_v == Approx(0.0).epsilon(EPS));
#     endif
    }
  }


	/* SECTION ("(Approximated) Lambda Operator") */
///	  {
////     // Check the definition of the Lambda operator (u = Lambda[S]) holds.
///
////     for (long m = 0; m < ndep; m++)
////     {
//// 			CHECK (relative_error (u[m][f], (Lambda[0]*SS)(m)) == Approx(0.0).epsilon(EPS));
//// 			CHECK (relative_error (v[m][f], (Lambda[0]*SS)(m)) == Approx(0.0).epsilon(EPS));
////     }
///		}

}



////////////////////////////////////////////////

TEST_CASE ("Analytic model")
{

  const long ndep  = 24;
  const long ndiag = ndep;

  vReal2 U;
  vReal2 V;
  vReal3 Ibdy;
  Long1 c2b;

  RAYPAIR raypair (ndep, 1, 0, 1, 0, U, V, Ibdy, c2b);

  vReal      S[ndep];
  vReal   dtau[ndep];
  vReal    tau[ndep];
  vReal      u[ndep];
  vReal      v[ndep];
  vReal Lambda[ndep];

  double S_local, dtau_local, u_sol;


  // Set up test data

  double DT = 0.01984687161267688;
  double SS = 1.1704086088847389e-16;  
  double BB = 1.5e-16;  

  for (long i = 0; i < ndep; i++)
  {
       u[i] = SS;
       v[i] = 0.0;
    dtau[i] = DT;
     tau[i] = 0.0;
  }

  for (long i = 0; i < ndep; i++)
  {
      raypair.Su[i] = SS;
      raypair.Sv[i] = 0.0;
    raypair.dtau[i] = DT;
             tau[i] = 0.0;
  }

  raypair.Su[0]      +=  2.0 / raypair.dtau[0]      * BB;
  raypair.Su[ndep-1] +=  2.0 / raypair.dtau[ndep-1] * BB;

  raypair.Sv[0]      +=  2.0 / raypair.dtau[0]      * (+BB-SS);
  raypair.Sv[ndep-1] +=  2.0 / raypair.dtau[ndep-1] * (-BB+SS);

  for (long i = 1; i < ndep; i++)
  {
    tau[i] = tau[i-1] + raypair.dtau[i];
  }

  raypair.ndep = ndep;

  raypair.solve ();

  solve_ray (ndep, u, v, dtau, ndiag, Lambda, ndep);


  SECTION ("Check with analytic solution")
  {

    for (long m = 0; m < ndep; m++)
    {
      vReal u_analytic = SS + 0.5*(BB-SS)*(exp(-tau[m]) + exp(tau[m]-tau[ndep-1]));
      vReal v_analytic =      0.5*(BB-SS)*(exp(-tau[m]) - exp(tau[m]-tau[ndep-1]));


      vReal error_u = relative_error(u_analytic, raypair.Su[m]);
      vReal error_v = relative_error(v_analytic, raypair.Sv[m]);

      cout << v_analytic << "\t" << raypair.Sv[m] << "\t" << error_v << endl;
    //  cout << relative_error(u[m], u[ndep-m]) << endl;
    //  cout << v_analytic << "\t" << v[m] << "\t" << error_v << endl;

#     if (GRID_SIMD)
        for (int lane = 0; lane < n_simd_lanes; lane++)
        {
          CHECK (error_u.getlane(lane) == Approx(0.0).epsilon(EPS));
          CHECK (error_v.getlane(lane) == Approx(0.0).epsilon(EPS));
        }
#     else
        CHECK (error_u == Approx(0.0).epsilon(EPS));
        CHECK (error_v == Approx(0.0).epsilon(EPS));
#     endif
    }
  }

}
