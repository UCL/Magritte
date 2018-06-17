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


int print (string text, vDouble a)
{
	for (int lane = 0; lane < n_vector_lanes; lane++)
	{
    cout << text << " " << a.getlane(lane) << endl;    
	}

	return (0);
}


vDouble vRelative_error (const vDouble a, const vDouble b)
{
	return (a-b) / (a+b);
}



///  feautrier_error: returns difference between the two sides of the Feautrier eq.
///    @param[in] i: index of point where Feautrier equation is to be evaluated
///////////////////////////////////////////////////////////////////////////////////

vDouble feautrier_error (const vDouble2& S, const vDouble2& dtau, const vDouble2& u,
		                     const long i, const long f)
{
 

  // Left hand side of Feautrier equation (d^2u / dtau^2 - u)

  vDouble lhs =  ( (u[i+1][f] - u[i][f])/dtau[i+1][f] - (u[i][f] - u[i-1][f])/dtau[i][f] )
		             / ( 0.5 * (dtau[i+1][f] + dtau[i][f]) ) - u[i][f];

  // Right hand side of Feautrier equation (-S)

  vDouble rhs = -S[i][f];


  return vRelative_error (rhs, lhs);

}




TEST_CASE ("Feautrier solver on feautrier1.txt")
{

  // Reading test data

  std::ifstream infile ("test_data/feautrier1.txt");

  const long ndep      = 100;
  const long nfreq     = n_vector_lanes;
	const long nfreq_red = 1;


  vDouble2     S (ndep, vDouble1 (nfreq_red));
  vDouble2  dtau (ndep, vDouble1 (nfreq_red));
	

  long   n;
	double S_local;
	double dtau_local;
	double u_sol;


  for (long i = 0; i < ndep; i++)
  {
  	infile >> n >> S_local >> dtau_local >> u_sol;

		   S[i][0] =    S_local;
		dtau[i][0] = dtau_local;
  	
  }


	//VectorXd SS (ndep);

  //for (long i = 0; i < ndep; i++)
  //{
  //	SS(i) = S[i][f];
  //}


  vDouble2      u (ndep, vDouble1 (nfreq_red));
  vDouble2      v (ndep, vDouble1 (nfreq_red));
	vDouble2 u_prev (ndep, vDouble1 (nfreq_red));
	vDouble2 v_prev (ndep, vDouble1 (nfreq_red));

	//vector <MatrixXd> Lambda (nfreq_red);
	//
  //MatrixXd temp (ndep,ndep);

	//Lambda[f] = temp;

	vDouble2 Lambda (ndep, vDouble1 (nfreq_red));
	

	long ndiag = ndep;


  for (long n = 0; n <= ndep; n++)
  {
		long n_r  = n;
		long n_ar = ndep-n;

    vDouble2   Su_r (n_r, vDouble1 (nfreq_red));
    vDouble2   Sv_r (n_r, vDouble1 (nfreq_red));
    vDouble2 dtau_r (n_r, vDouble1 (nfreq_red));

    vDouble2   Su_ar (n_ar, vDouble1 (nfreq_red));
    vDouble2   Sv_ar (n_ar, vDouble1 (nfreq_red));
    vDouble2 dtau_ar (n_ar, vDouble1 (nfreq_red));


    for (long m = 0; m < n_ar; m++)
    {
	  	for (long f = 0; f < nfreq_red; f++)
	  	{
    	    Su_ar[m][f] =    S[n_ar-1-m][f];
          Sv_ar[m][f] =    S[n_ar-1-m][f];
        dtau_ar[m][f] = dtau[n_ar-1-m][f];
	  	}
	  }	
	  	

    for (long m = 0; m < n_r; m++)
	  {
	  	for (long f = 0; f < nfreq_red; f++)
	  	{	
    	    Su_r[m][f] =    S[n_ar+m][f];
          Sv_r[m][f] =    S[n_ar+m][f];
        dtau_r[m][f] = dtau[n_ar+m][f];
	  	}
	  }


    solve_ray (n_r,  Su_r,  Sv_r,  dtau_r,
				       n_ar, Su_ar, Sv_ar, dtau_ar,
				       ndep, nfreq_red, u, v, ndiag, Lambda);


		/* SECTION ("Feautrier equation") */
		{
			// Check if result satisfies the Feautrier equation (d^2u/dtau^2=u-S)

      for (long m = 1; m < ndep-1; m++)
      {
				vDouble error_u = feautrier_error (S, dtau, u, m, 0);
        vDouble error_v = feautrier_error (S, dtau, v, m, 0);

	  		for (int lane = 0; lane < n_vector_lanes; lane++)
				{
          CHECK (error_u.getlane(lane) == Approx(0.0).epsilon(EPS));
          CHECK (error_v.getlane(lane) == Approx(0.0).epsilon(EPS));
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
    		{
				  vDouble error_u = vRelative_error (u[m][0], u_prev[m][0]);
          vDouble error_v = vRelative_error (v[m][0], v_prev[m][0]);

	  		  for (int lane = 0; lane < n_vector_lanes; lane++)
				  {
            CHECK (error_u.getlane(lane) == Approx(0.0).epsilon(EPS));
            CHECK (error_v.getlane(lane) == Approx(0.0).epsilon(EPS));
				  }
  			}
      }
  		
  		for (long m = 0; m < ndep; m++)
  		{
  			u_prev[m][0] = u[m][0];
  			v_prev[m][0] = v[m][0];
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
