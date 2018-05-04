// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <string>
#include <fstream>
#include <iostream>
#include <Eigen/Dense>

#include "catch.hpp"

#include "../src/feautrier.hpp"

#define EPS 1.0E-9


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

  return 2.0 * fabs(lhs - rhs) / fabs(lhs + rhs);

}



TEST_CASE ("Feautrier solver on feautrier1.txt")
{
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


  SECTION ("Feautrier u (mean) intensity")
  {
    double *u = new double[ndep];
    Eigen::MatrixXd Lambda(ndep,ndep);

    feautrier (ndep, S, dtau, u, Lambda);

    for (long i = 0; i < ndep; i++)
    {
      CHECK (Approx(u_sol[i]).epsilon(EPS) == u[i]);
    }

    for (long i = 0; i < ndep; i++)
    {
      // std::cout << u[i] << " " << u_sol[i] << std::endl;
    }

    for (long i = 1; i < ndep-1; i++)
    {
      double error1 = feautrier_error (i, S, dtau, u);

      // std::cout << "Our   error: " << error1 << " value " << u[i] << std::endl;

      double error2 = feautrier_error (i, S, dtau, u_sol);

      // std::cout << "Their error: " << error2 << " value " << u_sol[i] << std::endl;
    }

    delete [] u;
    delete [] L_diag_approx;

  }

  SECTION ("Approximate Lambda operator")
  {

    Eigen::MatrixXd Lambda(ndep,ndep);

  }

  delete [] S;
  delete [] dtau;
  delete [] u_sol;

}



TEST_CASE ("Feautrier solver on feautrier2.txt")
{
  std::ifstream infile ("test_data/feautrier2.txt");

  const long ndep = 100;
  long n;

  double *S     = new double[ndep];
  double *dtau  = new double[ndep];
  double *u_sol = new double[ndep];

  for (long i = 0; i < ndep; i++)
  {
    infile >> n >> S[i] >> dtau[i] >> u_sol[i];
  }


  SECTION ("Run on feautrier")
  {
    double *u             = new double[ndep];
    double *L_diag_approx = new double[ndep];

    feautrier (ndep, S, dtau, u, L_diag_approx);

    for (long i = 0; i < ndep; i++)
    {
      CHECK (Approx(u_sol[i]).epsilon(EPS) == u[i]);
    }

    for (long i = 0; i < ndep; i++)
    {
      // std::cout << u[i] << " " << u_sol[i] << std::endl;
    }

    for (long i = 1; i < ndep-1; i++)
    {
      double error1 = feautrier_error (i, S, dtau, u);

      // std::cout << "Our   error: " << error1 << std::endl;

      double error2 = feautrier_error (i, S, dtau, u_sol);

      // std::cout << "Their error: " << error2 << std::endl;
    }

    delete [] u;
    delete [] L_diag_approx;

  }


  delete [] S;
  delete [] dtau;
  delete [] u_sol;

}
