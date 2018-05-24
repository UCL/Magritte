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

double feautrier_error (long i, vector<double>& S, vector<double>& dtau, vector<double>& u)
{

  // Left hand side of Feautrier equation (d^2u / dtau^2 - u)

  double lhs =  ( (u[i+1] - u[i])/dtau[i+1] - (u[i] - u[i-1])/dtau[i] )
                / ( (dtau[i+1] + dtau[i]) / 2.0 ) - u[i];

  // Right hand side of Feautrier equation (-S)

  double rhs = -S[i];


  return relative_error (rhs, lhs);

}




TEST_CASE ("Feautrier solver on feautrier1.txt")
{

  // Reading test data

  std::ifstream infile ("test_data/feautrier1.txt");

  const long ndep = 100;

  long n;

  vector<double>     S (ndep);
  vector<double>  dtau (ndep);
  vector<double> u_sol (ndep);
	

  for (long i = 0; i < ndep; i++)
  {
  	infile >> n >> S[i] >> dtau[i] >> u_sol[i];
  }


  vector<double>      u (ndep);
  vector<double>      v (ndep);
	vector<double> u_prev (ndep);
	vector<double> v_prev (ndep);

	vector <MatrixXd> Lambda (1);
	
  MatrixXd temp (ndep,ndep);

	Lambda[0] = temp;
	

	long ndiag = ndep;


  for (long n = 0; n <= ndep; n++)
  {
		long n_r  = n;
		long n_ar = ndep-n;

    vector<double>   Su_r (n_r);
    vector<double>   Sv_r (n_r);
    vector<double> dtau_r (n_r);

    vector<double>   Su_ar (n_ar);
    vector<double>   Sv_ar (n_ar);
    vector<double> dtau_ar (n_ar);


    for (long m = 0; m < n_ar; m++)
		{
  		  Su_ar[m] =    S[n_ar-1-m];
  		  Sv_ar[m] =    S[n_ar-1-m];
			dtau_ar[m] = dtau[n_ar-1-m];
		}	
		
    for (long m = 0; m < n_r; m++)
		{
  		  Su_r[m] =    S[n_ar+m];
  		  Sv_r[m] =    S[n_ar+m];
			dtau_r[m] = dtau[n_ar+m];
		}


    solve_ray (n_r,  Su_r,  Sv_r,  dtau_r,
				       n_ar, Su_ar, Sv_ar, dtau_ar,
				       ndep, 1, u, v, ndiag, Lambda);


		/* SECTION ("Feautrier equation") */
		{
			// Check if result satisfies the Feautrier equation (d^2u/dtau^2=u-S)

      for (long m = 1; m < ndep-1; m++)
      {
        CHECK (feautrier_error (m, S, dtau, u) == Approx(0.0).epsilon(EPS));
        CHECK (feautrier_error (m, S, dtau, v) == Approx(0.0).epsilon(EPS));
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
  		  	CHECK ( relative_error(u_prev[m], u[m]) == Approx(0.0).epsilon(EPS));
  		  	CHECK ( relative_error(v_prev[m], v[m]) == Approx(0.0).epsilon(EPS));
  			}
      }
  		
  		for (long m = 0; m < ndep; m++)
  		{
  			u_prev[m] = u[m];
  			v_prev[m] = v[m];
  		}
		}  


  	/* SECTION ("(Approximated) Lambda Operator") */
	  {	
      // Check the definition of the Lambda operator (u = Lambda[S]) holds.   

			double *pu = &u[0];
      Map<VectorXd> uu (pu,ndep);
			double *pv = &v[0];
      Map<VectorXd> vv (pv,ndep);
			double *pS = &S[0];
      Map<VectorXd> SS (pS,ndep);
  
      for (long m = 0; m < ndep; m++)
      {
  			CHECK (relative_error (uu(m), (Lambda[0]*SS)(m)) == Approx(0.0).epsilon(EPS));
  			CHECK (relative_error (vv(m), (Lambda[0]*SS)(m)) == Approx(0.0).epsilon(EPS));
      }
		}

  }


}
