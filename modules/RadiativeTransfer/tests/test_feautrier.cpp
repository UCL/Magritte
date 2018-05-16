// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <string>
#include <fstream>
#include <iostream>
#include <Eigen/Core>
#include <Eigen/Dense>

#include "catch.hpp"

#include "../src/feautrier.cpp"

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

double feautrier_error (long i, double *S, double *dtau, double *u)
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

  double *S     = new double[ndep];
  double *dtau  = new double[ndep];
  double *u_sol = new double[ndep];
	

  for (long i = 0; i < ndep; i++)
  {
  	infile >> n >> S[i] >> dtau[i] >> u_sol[i];
  }


  double *u      = new double[ndep];
	double *u_prev = new double[ndep];

  Eigen::MatrixXd Lambda (ndep,ndep);
	

	long ndiag = ndep;


  for (long n = 0; n <= ndep; n++)
  {
		long n_r  = n;
		long n_ar = ndep-n;

    double     *S_r = new double[n_r];
    double  *dtau_r = new double[n_r];

    double    *S_ar = new double[n_ar];
    double *dtau_ar = new double[n_ar];


    for (long m = 0; m < n_ar; m++)
		{
  		   S_ar[m] =    S[n_ar-1-m];
			dtau_ar[m] = dtau[n_ar-1-m];
		}	
		
    for (long m = 0; m < n_r; m++)
		{
  		   S_r[m] =    S[n_ar+m];
			dtau_r[m] = dtau[n_ar+m];
		}


    feautrier (n_r, S_r, dtau_r, n_ar, S_ar, dtau_ar, u, Lambda, ndiag);


		/* SECTION ("Feautrier equation") */
		{
			// Check if result satisfies the Feautrier equation (d^2u/dtau^2=u-S)

      for (long m = 1; m < ndep-1; m++)
      {
        CHECK (feautrier_error (m, S, dtau, u) == Approx(0.0).epsilon(EPS));
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
  			}
      }
  		
  		for (long m = 0; m < ndep; m++)
  		{
  			u_prev[m] = u[m];
  		}
		}  


  	/* SECTION ("(Approximated) Lambda Operator") */
	  {	
      // Check the definition of the Lambda operator (u = Lambda[S]) holds.   

      Eigen::Map <Eigen::VectorXd> uu (u,ndep);
      Eigen::Map <Eigen::VectorXd> SS (S,ndep);
  
      for (long m = 0; m < ndep; m++)
      {
  			CHECK (relative_error (uu(m), (Lambda*SS)(m)) == Approx(0.0).epsilon(EPS));
      }
		}

		
   	delete [] S_r;
  	delete [] dtau_r;
  	delete [] S_ar;
  	delete [] dtau_ar;
  }


  delete [] u;
  delete [] u_prev;

  delete [] S;
  delete [] dtau;
  delete [] u_sol;

}
