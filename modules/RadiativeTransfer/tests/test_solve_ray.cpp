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

#include "../src/solve_ray.hpp"

#define EPS 1.0E-4


///  relative_error: returns relative error between A and B
///////////////////////////////////////////////////////////

double relative_error (double A, double B)
{
  return 2.0 * fabs(A-B) / fabs(A+B);
}




///  feautrier_error: returns difference between the two sides of the Feautrier eq.
///    @param[in] i: index of point where Feautrier equation is to be evaluated
///////////////////////////////////////////////////////////////////////////////////

double feautrier_error (long i, long f, vector<vector<double>> S, vector<vector<double>> dtau, vector<vector<double>> u)
{

  // Left hand side of Feautrier equation (d^2u / dtau^2 - u)

  double lhs =  ( (u[i+1][f] - u[i][f])/dtau[i+1][f] - (u[i][f] - u[i-1][f])/dtau[i][f] )
                / ( (dtau[i+1][f] + dtau[i][f]) / 2.0 ) - u[i][f];

  // Right hand side of Feautrier equation (-S)

  double rhs = -S[i][f];


  return relative_error (rhs, lhs);

}




TEST_CASE ("Feautrier solver on feautrier1.txt")
{

  // Reading test data

  std::ifstream infile ("test_data/feautrier1.txt");

  const long ndep = 100;
  const long nfreq = 1;

  long n;
	long f = 0;

  vector<vector<double>>     S (ndep, vector<double> (nfreq));
  vector<vector<double>>  dtau (ndep, vector<double> (nfreq));
  vector<double>         u_sol (ndep);
	

  for (long i = 0; i < ndep; i++)
  {
  	infile >> n >> S[i][f] >> dtau[i][f] >> u_sol[i];
  }


	VectorXd SS (ndep);

  for (long i = 0; i < ndep; i++)
  {
  	SS(i) = S[i][f];
  }


  vector<vector<double>>      u (ndep, vector<double> (nfreq));
  vector<vector<double>>      v (ndep, vector<double> (nfreq));
	vector<vector<double>> u_prev (ndep, vector<double> (nfreq));
	vector<vector<double>> v_prev (ndep, vector<double> (nfreq));

	vector <MatrixXd> Lambda (nfreq);
	
  MatrixXd temp (ndep,ndep);

	Lambda[f] = temp;
	

	long ndiag = ndep;


  for (long n = 0; n <= ndep; n++)
  {
		long n_r  = n;
		long n_ar = ndep-n;

    vector<vector<double>>   Su_r (n_r, vector<double> (nfreq));
    vector<vector<double>>   Sv_r (n_r, vector<double> (nfreq));
    vector<vector<double>> dtau_r (n_r, vector<double> (nfreq));

    vector<vector<double>>   Su_ar (n_ar, vector<double> (nfreq));
    vector<vector<double>>   Sv_ar (n_ar, vector<double> (nfreq));
    vector<vector<double>> dtau_ar (n_ar, vector<double> (nfreq));


    for (long m = 0; m < n_ar; m++)
		{
  		  Su_ar[m][f] =    S[n_ar-1-m][f];
  		  Sv_ar[m][f] =    S[n_ar-1-m][f];
			dtau_ar[m][f] = dtau[n_ar-1-m][f];
		}	
		
    for (long m = 0; m < n_r; m++)
		{
  		  Su_r[m][f] =    S[n_ar+m][f];
  		  Sv_r[m][f] =    S[n_ar+m][f];
			dtau_r[m][f] = dtau[n_ar+m][f];
		}


    solve_ray (n_r,  Su_r,  Sv_r,  dtau_r,
				       n_ar, Su_ar, Sv_ar, dtau_ar,
				       ndep, nfreq, u, v, ndiag, Lambda);


		/* SECTION ("Feautrier equation") */
		{
			// Check if result satisfies the Feautrier equation (d^2u/dtau^2=u-S)

      for (long m = 1; m < ndep-1; m++)
      {
        CHECK (feautrier_error (m, f, S, dtau, u) == Approx(0.0).epsilon(EPS));
        CHECK (feautrier_error (m, f, S, dtau, v) == Approx(0.0).epsilon(EPS));
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
  		  	CHECK (relative_error(u_prev[m][f], u[m][f]) == Approx(0.0).epsilon(EPS));
  		  	CHECK (relative_error(v_prev[m][f], v[m][f]) == Approx(0.0).epsilon(EPS));
  			}
      }
  		
  		for (long m = 0; m < ndep; m++)
  		{
  			u_prev[m][f] = u[m][f];
  			v_prev[m][f] = v[m][f];
  		}
		}  


  	/* SECTION ("(Approximated) Lambda Operator") */
	  {	
      // Check the definition of the Lambda operator (u = Lambda[S]) holds.   
  
      for (long m = 0; m < ndep; m++)
      {
  			CHECK (relative_error (u[m][f], (Lambda[0]*SS)(m)) == Approx(0.0).epsilon(EPS));
  			CHECK (relative_error (v[m][f], (Lambda[0]*SS)(m)) == Approx(0.0).epsilon(EPS));
      }
		}

  }


}
