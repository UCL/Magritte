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

#include "solve_ray.hpp"
#include "GridTypes.hpp"
#include "folders.hpp"

#define EPS 1.0E-6


///  feautrier_error: returns difference between the two sides of the Feautrier eq.
///    @param[in] S: pointer to source function
///    @param[in] dtau: pointer to optical depth increments
///    @param[in] u: pointer to source function
///    @param[in] i: index of point where Feautrier equation is to be evaluated
///////////////////////////////////////////////////////////////////////////////////

vReal feautrier_error (const vReal* S, const vReal* dtau, const vReal* u, const long i)
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

		u[i] = v[i] = S[i] =    S_local;
		           dtau[i] = dtau_local;
  }


  solve_ray (ndep, u, v, dtau, ndiag, Lambda, ndep);


	SECTION ("Feautrier equation")
	{

		// Check if result satisfies the Feautrier equation (d^2u/dtau^2=u-S)

    for (long m = 1; m < ndep-1; m++)
    {
			vReal error_u = feautrier_error (S, dtau, u, m);
      vReal error_v = feautrier_error (S, dtau, v, m);


	 		for (int lane = 0; lane < n_simd_lanes; lane++)
			{
#        if (GRID_SIMD)
          CHECK (error_u.getlane(lane) == Approx(0.0).epsilon(EPS));
          CHECK (error_v.getlane(lane) == Approx(0.0).epsilon(EPS));
#				else
          CHECK (error_u == Approx(0.0).epsilon(EPS));
          CHECK (error_v == Approx(0.0).epsilon(EPS));
#        endif
			}
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
