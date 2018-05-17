// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <math.h>
#include <Eigen/Core>
#include "solve_ray.hpp"


///  solve_ray: solve radiative transfer equation using the Feautrier method
///  and the numerical scheme devised by Rybicki & Hummer (1991)
///    @param[in] n_r: number of points on ray r
///    @param[in] *Su_r: pointer to source function for u along ray r
///    @param[in] *Sv_r: pointer to source function for v along ray r
///    @param[in] *dtau_r: pointer to optical depth increments along ray r
///    @param[in] n_ar: number of points on ray ar
///    @param[in] *Su_ar: pointer to source function for u along ray ar
///    @param[in] *Sv_ar: pointer to source function for v along ray ar
///    @param[in] *dtau_ar: pointer to optical depth increments along ray ar
///    @param[in] ndep: total number of points on the combined ray r and ar
///    @param[out] *u: pointer to resulting Feautrier mean intensity vector
///    @param[out] *v: pointer to resulting Feautrier flux intensity vector
///    @param[in] ndiag: degree of approximation in ALO (e.g. 0->diag, 1->tridiag)
///    @param[out] Lambda: approximate Lambda operator (ALO) for this ray pair
//////////////////////////////////////////////////////////////////////////////////

int solve_ray (const long n_r,  double *Su_r,  double *Sv_r,  double *dtau_r,
	             const long n_ar, double *Su_ar, double *Sv_ar, double *dtau_ar,
	             const long ndep, double *u,     double *v,
							 const long ndiag, Eigen::Ref <Eigen::MatrixXd> Lambda)
{

	double B0;                      // B[0]
  double B0_min_C0;               // B[0] - C[0]
  double Bd;                      // B[ndep-1]
	double Bd_min_Ad;               // B[ndep-1] - A[ndep-1]

  double *A = new double[ndep];   // A coefficient in Feautrier recursion relation
  double *C = new double[ndep];   // C coefficient in Feautrier recursion relation
  double *F = new double[ndep];   // helper variable from Rybicki & Hummer (1991)
  double *G = new double[ndep];   // helper variable from Rybicki & Hummer (1991)




  // SETUP FEAUTRIER RECURSION RELATION
  // __________________________________


	if ( (n_ar > 0) && (n_r > 0) )
	{
    A[n_ar-1] = 2.0 / ((dtau_ar[0] + dtau_r[0]) * dtau_ar[0]);
    C[n_ar-1] = 2.0 / ((dtau_ar[0] + dtau_r[0]) * dtau_r[0]);
	}


	if (n_ar > 0)
	{
    A[0] = 0.0;
    C[0] = 2.0/(dtau_ar[n_ar-1]*dtau_ar[n_ar-1]);

    B0        = 1.0 + 2.0/dtau_ar[n_ar-1] + 2.0/(dtau_ar[n_ar-1]*dtau_ar[n_ar-1]);
    B0_min_C0 = 1.0 + 2.0/dtau_ar[n_ar-1];

    for (long n = n_ar-1; n > 1; n--)
    {
      A[n_ar-n] = 2.0 / ((dtau_ar[n-1] + dtau_ar[n-2]) * dtau_ar[n-1]);
      C[n_ar-n] = 2.0 / ((dtau_ar[n-1] + dtau_ar[n-2]) * dtau_ar[n-2]);
    }
	}


	if (n_r > 0)
	{
    for (long n = 0; n < n_r-1; n++)
		{
      A[n_ar+n] = 2.0 / ((dtau_r[n] + dtau_r[n+1]) * dtau_r[n]);
      C[n_ar+n] = 2.0 / ((dtau_r[n] + dtau_r[n+1]) * dtau_r[n+1]);
    }

    A[ndep-1] = 2.0/(dtau_r[n_r-1]*dtau_r[n_r-1]);
    C[ndep-1] = 0.0;

    Bd        = 1.0 + 2.0/dtau_r[n_r-1] + 2.0/(dtau_r[n_r-1]*dtau_r[n_r-1]);
    Bd_min_Ad = 1.0 + 2.0/dtau_r[n_r-1];
	}


	if (n_ar == 0)
	{
    A[0] = 0.0;
    C[0] = 2.0/(dtau_r[0]*dtau_r[0]);

    B0        = 1.0 + 2.0/dtau_r[0] + 2.0/(dtau_r[0]*dtau_r[0]);
    B0_min_C0 = 1.0 + 2.0/dtau_r[0];
	}


	if (n_r == 0)
	{
    A[ndep-1] = 2.0/(dtau_ar[0]*dtau_ar[0]);
    C[ndep-1] = 0.0;

    Bd        = 1.0 + 2.0/dtau_ar[0] + 2.0/(dtau_ar[0]*dtau_ar[0]);
    Bd_min_Ad = 1.0 + 2.0/dtau_ar[0];
	}


  // Store source function S initially in u

  for (long n = n_ar-1; n >= 0; n--)
  {
    u[n_ar-1-n] = Su_ar[n];
    v[n_ar-1-n] = Sv_ar[n];
  }

  for (long n = 0; n < n_r; n++)
  {
    u[n_ar+n] = Su_r[n];
    v[n_ar+n] = Sv_r[n];
  }




  // SOLVE FEAUTRIER RECURSION RELATION
  // __________________________________


  // Elimination step

  u[0] = u[0] / B0;
  v[0] = v[0] / B0;

  F[0] = B0_min_C0 / C[0];


  for (long n = 1; n < ndep-1; n++)
  {
    F[n] = (1.0 + A[n]*F[n-1]/(1.0 + F[n-1])) / C[n];

    u[n] = (u[n] + A[n]*u[n-1]) / ((1.0 + F[n]) * C[n]);
    v[n] = (v[n] + A[n]*v[n-1]) / ((1.0 + F[n]) * C[n]);
  }


  u[ndep-1] = (u[ndep-1] + A[ndep-1]*u[ndep-2])
              / (Bd_min_Ad + Bd*F[ndep-2]) * (1.0 + F[ndep-2]);

  v[ndep-1] = (v[ndep-1] + A[ndep-1]*v[ndep-2])
              / (Bd_min_Ad + Bd*F[ndep-2]) * (1.0 + F[ndep-2]);

  G[ndep-1] = Bd_min_Ad / A[ndep-1];


  // Back substitution

  for (long n = ndep-2; n > 0; n--)
  {
    u[n] = u[n] + u[n+1]/(1.0+F[n]);
    v[n] = v[n] + v[n+1]/(1.0+F[n]);

    G[n] = (1.0 + C[n]*G[n+1]/(1.0+G[n+1])) / A[n];
  }

  u[0] = u[0] + u[1]/(1.0+F[0]);
  v[0] = v[0] + v[1]/(1.0+F[0]);




  // CALCULATE LAMBDA OPERATOR
  // _________________________


  // Calculate diagonal elements

  Lambda(0,0) = (1.0 + G[1]) / (B0_min_C0 + B0*G[1]);

  for (long n = 1; n < ndep-1; n++)
  {
    Lambda(n,n) = (1.0 + G[n+1]) / ((F[n] + G[n+1] + F[n]*G[n+1]) * C[n]);
  }

  Lambda(ndep-1,ndep-1) = (1.0 + F[ndep-2]) / (Bd_min_Ad + Bd*F[ndep-2]);


  // Add upper-diagonal elements

  for (long m = 1; m < ndiag; m++)
  {	  
    for (long n = 0; n < ndep-m; n++)
    {
      Lambda(n,n+m) = Lambda(n+1,n+m) / (1.0 + F[n]);
    }
  }


  // Add lower-diagonal elements

  for (long m = 1; m < ndiag; m++)
  {	  
    for (long n = m; n < ndep; n++)
    {
      Lambda(n,n-m) = Lambda(n-1,n-m) / (1.0 + G[n]);
    }
  }


  // Free allocated memory for temporary variables

  delete [] A;
  delete [] C;
  delete [] F;
  delete [] G;


  return (0);

}
