// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <vector>
using namespace std;
#include <Eigen/Core>
using namespace Eigen;

#include "solve_ray.hpp"

#define c ( nfreq*(c) + (f) )

///  solve_ray: solve radiative transfer equation using the Feautrier method
///  and the numerical scheme devised by Rybicki & Hummer (1991)
///    @param[in] n_r: number of points on ray r
///    @param[in] Su_r: reference to source function for u along ray r
///    @param[in] Sv_r: reference to source function for v along ray r
///    @param[in] dtau_r: reference to optical depth increments along ray r
///    @param[in] n_ar: number of points on ray ar
///    @param[in] Su_ar: reference to source function for u along ray ar
///    @param[in] Sv_ar: reference to source function for v along ray ar
///    @param[in] dtau_ar: reference to optical depth increments along ray ar
///    @param[in] ndep: total number of points on the combined ray r and ar
///    @param[in] nfreq: number of frequency bins
///    @param[out] u: reference to resulting Feautrier mean intensity vector
///    @param[out] v: reference to resulting Feautrier flux intensity vector
///    @param[in] ndiag: degree of approximation in ALO (e.g. 0->diag, 1->tridiag)
///    @param[out] Lambda: approximate Lambda operator (ALO) for this ray pair
//////////////////////////////////////////////////////////////////////////////////

int solve_ray (const long n_r,  vector<VectorXd>& Su_r,  vector<VectorXd>& Sv_r,  vector<VectorXd>& dtau_r,
	             const long n_ar, vector<VectorXd>& Su_ar, vector<VectorXd>& Sv_ar, vector<VectorXd>& dtau_ar,
	             const long ndep, const long nfreq,        vector<VectorXd>& u,     vector<VectorXd>& v,
							 const long ndiag, vector<MatrixXd>& Lambda)
{

	VectorXd B0        (nfreq);   // B[0]
  VectorXd B0_min_C0 (nfreq);   // B[0] - C[0]
  VectorXd Bd        (nfreq);   // B[ndep-1]
	VectorXd Bd_min_Ad (nfreq);   // B[ndep-1] - A[ndep-1]

  vector<VectorXd> A (ndep);   // A coefficient in Feautrier recursion relation
	vector<VectorXd> C (ndep);   // C coefficient in Feautrier recursion relation
  vector<VectorXd> F (ndep);   // helper variable from Rybicki & Hummer (1991)
  vector<VectorXd> G (ndep);   // helper variable from Rybicki & Hummer (1991)

	for (long f = 0; f < nfreq; f++)
	{
    A[f].reserve(nfreq); 
    C[f].reserve(nfreq); 
    F[f].reserve(nfreq); 
    G[f].reserve(nfreq); 
	}



		
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

  Lambda[f](0,0) = (1.0 + G[1]) / (B0_min_C0[f] + B0[f]*G[1]);

  for (long n = 1; n < ndep-1; n++)
  {
    for (long f = 0; f < nfreq; f++)
		{
      Lambda[f](n,n) = (1.0 + G[n+1]) / ((F[n] + G[n+1] + F[n]*G[n+1]) * C[n]);
		}
  }

  for (long f = 0; f < nfreq; f++)
	{
    Lambda[f](ndep-1,ndep-1) = (1.0 + F[ndep-2]) / (Bd_min_Ad[f] + Bd[f]*F[ndep-2]);
	}

  // Add upper-diagonal elements

  for (long m = 1; m < ndiag; m++)
  {	  
    for (long n = 0; n < ndep-m; n++)
    {
      for (long f = 0; f < nfreq; f++)
			{
        Lambda[f](n,n+m) = Lambda[f](n+1,n+m) / (1.0 + F[n]);
			}
    }
  }


  // Add lower-diagonal elements

  for (long m = 1; m < ndiag; m++)
  {	  
    for (long n = m; n < ndep; n++)
    {
      for (long f = 0; f < nfreq; f++)
      {    
        Lambda[f](n,n-m) = Lambda[f](n-1,n-m) / (1.0 + G[n]);
		  }
    }
  }


  return (0);

}
