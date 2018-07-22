// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <fstream>
#include <iostream>
#include <vector>
using namespace std;
#include <Eigen/Core>
using namespace Eigen;

#include "catch.hpp"
#include "tools.hpp"

#include "../src/solve_ray.hpp"
#include "../src/types.hpp"
#include "../src/GridTypes.hpp"

#define EPS 1.0E-4


int print (string text, vReal a)
{
	for (int lane = 0; lane < 1/*n_simd_lanes*/; lane++)
	{
    cout << text << " " << a.getlane(lane) << endl;
	}

	return (0);
}


vReal vRelative_error (const vReal a, const vReal b)
{
	return (a-b) / (a+b);
}



///  feautrier_error: returns difference between the two sides of the Feautrier eq.
///    @param[in] i: index of point where Feautrier equation is to be evaluated
///////////////////////////////////////////////////////////////////////////////////

vReal feautrier_error (const vReal& S, const vReal& dtau, const vReal& u,
	                     const long i, const long f)
{


  // Left hand side of Feautrier equation (d^2u / dtau^2 - u)

  vReal lhs =  ( (u[i+1] - u[i])/dtau[i+1] - (u[i] - u[i-1])/dtau[i] )
		            / ( 0.5 * (dtau[i+1] + dtau[i]) ) - u[i];

  // Right hand side of Feautrier equation (-S)

  vReal rhs = -S[i];


  return vRelative_error (rhs, lhs);

}




TEST_CASE ("Feautrier solver on feautrier1.txt")
{

  // Reading test data

  ifstream infile ("test_data/feautrier1.txt");

  const long ndep      = 100;
  const long nfreq     = n_simd_lanes;
	const long nfreq_red = 1;


  vReal1     S (ndep);
  vReal1  dtau (ndep);


	double S_local;
	double dtau_local;
	double u_sol;


  for (long i = 0; i < ndep; i++)
  {
  	infile >> S_local >> dtau_local >> u_sol;

		   S[i] =    S_local;
		dtau[i] = dtau_local;

		//print ("S = ", S[i][0]);
		//print ("d = ", dtau[i][0]);
  }


	//VectorXd SS (ndep);

  //for (long i = 0; i < ndep; i++)
  //{
  //	SS(i) = S[i][f];
  //}


  vReal      u [ndep];
  vReal      v [ndep];
	vReal u_prev [ndep];
	vReal v_prev [ndep];

  vReal      A [ndep];
  vReal      C [ndep];
  vReal      F [ndep];
  vReal      G [ndep];

	vReal B0       ;
  vReal B0_min_C0;
  vReal Bd       ;
	vReal Bd_min_Ad;

	//vector <MatrixXd> Lambda (nfreq_red);
	//
  //MatrixXd temp (ndep,ndep);

	//Lambda[f] = temp;

	vReal Lambda [ndep];


	long ndiag = ndep;


  for (long n = 0; n <= ndep; n++)
  {
		long n_r  = n;
		long n_ar = ndep-n;

    vReal   Su_r[n_r];
    vReal   Sv_r[n_r];
    vReal dtau_r[n_r];

    vReal   Su_ar[n_ar];
    vReal   Sv_ar[n_ar];
    vReal dtau_ar[n_ar];


    for (long m = 0; m < n_ar; m++)
    {
			if (n_ar > 0)
			{
    	    Su_ar[m] =    S[n_ar-1-m];
          Sv_ar[m] =    S[n_ar-1-m];
        dtau_ar[m] = dtau[n_ar-1-m];
			}
	  }


    for (long m = 0; m < n_r; m++)
	  {
			if (n_r > 0)
			{
    	    Su_r[m] =    S[n_ar+m];
          Sv_r[m] =    S[n_ar+m];
        dtau_r[m] = dtau[n_ar+m];
			}
	  }


		for (long f = 0; f < nfreq_red; f++)
		{
//      solve_ray (n_r,  Su_r,  Sv_r,  dtau_r,
//		  		       n_ar, Su_ar, Sv_ar, dtau_ar,
//								 A, C, F, G,
//								 B0, B0_min_C0, Bd, Bd_min_Ad,
//		  		       ndep, u, v, ndiag, Lambda);
		}


		/* SECTION ("Feautrier equation") */
		{
			// Check if result satisfies the Feautrier equation (d^2u/dtau^2=u-S)

      for (long m = 1; m < ndep-1; m++)
      {
				vReal error_u = feautrier_error (S, dtau, u, m, 0);
        vReal error_v = feautrier_error (S, dtau, v, m, 0);

				//print ("u = ", u[m]);
				//print ("d = ", dtau[m][0]);

	  		for (int lane = 0; lane < n_simd_lanes; lane++)
				{
#         if (GRID_SIMD)
            CHECK (error_u.getlane(lane) == Approx(0.0).epsilon(EPS));
            CHECK (error_v.getlane(lane) == Approx(0.0).epsilon(EPS));
#					else
            CHECK (error_u == Approx(0.0).epsilon(EPS));
            CHECK (error_v == Approx(0.0).epsilon(EPS));
#         endif
				}
      }
		}


		/* SECTION ("Translation invariance along ray") */
		{
			// Check if the result is indendent of the posiion of the origin
			// (i.e. check indenpendece of division over ray r and ray ar).

      for (long m = 1; m < ndep-1; m++)
      {
   			if (n != 0)
   			//if (n == 1)
    		{
				  vReal error_u = vRelative_error (u[m], u_prev[m]);
          vReal error_v = vRelative_error (v[m], v_prev[m]);

	  		  for (int lane = 0; lane < n_simd_lanes; lane++)
				  {
#         if (GRID_SIMD)
            CHECK (error_u.getlane(lane) == Approx(0.0).epsilon(EPS));
            CHECK (error_v.getlane(lane) == Approx(0.0).epsilon(EPS));
#					else
            CHECK (error_u == Approx(0.0).epsilon(EPS));
            CHECK (error_v == Approx(0.0).epsilon(EPS));
#         endif
				  }
  			}
      }

  		for (long m = 0; m < ndep; m++)
  		{
  			u_prev[m] = u[m];
  			v_prev[m] = v[m];
  		}
		}


  	/* SECTION ("(Approximated) Lambda Operator") */
//	  {
//      // Check the definition of the Lambda operator (u = Lambda[S]) holds.
//
//      for (long m = 0; m < ndep; m++)
//      {
//  			CHECK (relative_error (u[m][f], (Lambda[0]*SS)(m)) == Approx(0.0).epsilon(EPS));
//  			CHECK (relative_error (v[m][f], (Lambda[0]*SS)(m)) == Approx(0.0).epsilon(EPS));
//      }
//		}

  }


}
