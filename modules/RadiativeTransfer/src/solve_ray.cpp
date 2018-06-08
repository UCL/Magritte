// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <vector>
using namespace std;
#include <Eigen/Core>
using namespace Eigen;

#include "solve_ray.hpp"


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

int solve_ray (const long n_r,  vector<vector<double>>& Su_r,  vector<vector<double>>& Sv_r,  vector<vector<double>>& dtau_r,
	             const long n_ar, vector<vector<double>>& Su_ar, vector<vector<double>>& Sv_ar, vector<vector<double>>& dtau_ar,
	             const long ndep, const long nfreq,              vector<vector<double>>& u,     vector<vector<double>>& v,
							 const long ndiag, vector<MatrixXd>& Lambda)
{

	vector<double> B0        (nfreq);   // B[0][f]
  vector<double> B0_min_C0 (nfreq);   // B[0][f] - C[0][f]
  vector<double> Bd        (nfreq);   // B[ndep-1][f]
	vector<double> Bd_min_Ad (nfreq);   // B[ndep-1][f] - A[ndep-1][f]

  vector<vector<double>> A (ndep, vector<double> (nfreq));   // A coefficient in Feautrier recursion relation
	vector<vector<double>> C (ndep, vector<double> (nfreq));   // C coefficient in Feautrier recursion relation
  vector<vector<double>> F (ndep, vector<double> (nfreq));   // helper variable from Rybicki & Hummer (1991)
  vector<vector<double>> G (ndep, vector<double> (nfreq));   // helper variable from Rybicki & Hummer (1991)




  // SETUP FEAUTRIER RECURSION RELATION
  // __________________________________


	if ( (n_ar > 0) && (n_r > 0) )
	{
		for (long f = 0; f < nfreq; f++)
		{
      A[n_ar-1][f] = 2.0 / ((dtau_ar[0][f] + dtau_r[0][f]) * dtau_ar[0][f]);
      C[n_ar-1][f] = 2.0 / ((dtau_ar[0][f] + dtau_r[0][f]) * dtau_r[0][f]);
		}
	}


	if (n_ar > 0)
	{
		for (long f = 0; f < nfreq; f++)
		{
      A[0][f] = 0.0;
      C[0][f] = 2.0/(dtau_ar[n_ar-1][f]*dtau_ar[n_ar-1][f]);

      B0       [f] = 1.0 + 2.0/dtau_ar[n_ar-1][f] + 2.0/(dtau_ar[n_ar-1][f]*dtau_ar[n_ar-1][f]);
      B0_min_C0[f] = 1.0 + 2.0/dtau_ar[n_ar-1][f];
		}

    for (long n = n_ar-1; n > 1; n--)
    {
		  for (long f = 0; f < nfreq; f++)
			{		
        A[n_ar-n][f] = 2.0 / ((dtau_ar[n-1][f] + dtau_ar[n-2][f]) * dtau_ar[n-1][f]);
        C[n_ar-n][f] = 2.0 / ((dtau_ar[n-1][f] + dtau_ar[n-2][f]) * dtau_ar[n-2][f]);
			}
    }
	}


	if (n_r > 0)
	{
    for (long n = 0; n < n_r-1; n++)
		{
		  for (long f = 0; f < nfreq; f++)
			{
        A[n_ar+n][f] = 2.0 / ((dtau_r[n][f] + dtau_r[n+1][f]) * dtau_r[n][f]);
        C[n_ar+n][f] = 2.0 / ((dtau_r[n][f] + dtau_r[n+1][f]) * dtau_r[n+1][f]);
			}
    }

	  for (long f = 0; f < nfreq; f++)
		{
      A[ndep-1][f] = 2.0/(dtau_r[n_r-1][f]*dtau_r[n_r-1][f]);
      C[ndep-1][f] = 0.0;

      Bd       [f] = 1.0 + 2.0/dtau_r[n_r-1][f] + 2.0/(dtau_r[n_r-1][f]*dtau_r[n_r-1][f]);
      Bd_min_Ad[f] = 1.0 + 2.0/dtau_r[n_r-1][f];
		}
	}


	if (n_ar == 0)
	{
	  for (long f = 0; f < nfreq; f++)
		{
      A[0][f] = 0.0;
      C[0][f] = 2.0/(dtau_r[0][f]*dtau_r[0][f]);

      B0       [f] = 1.0 + 2.0/dtau_r[0][f] + 2.0/(dtau_r[0][f]*dtau_r[0][f]);
      B0_min_C0[f] = 1.0 + 2.0/dtau_r[0][f];
		}
	}


	if (n_r == 0)
	{
	  for (long f = 0; f < nfreq; f++)
		{
      A[ndep-1][f] = 2.0/(dtau_ar[0][f]*dtau_ar[0][f]);
      C[ndep-1][f] = 0.0;

      Bd       [f] = 1.0 + 2.0/dtau_ar[0][f] + 2.0/(dtau_ar[0][f]*dtau_ar[0][f]);
      Bd_min_Ad[f] = 1.0 + 2.0/dtau_ar[0][f];
		}
	}


  // Store source function S initially in u

  for (long n = n_ar-1; n >= 0; n--)
  {
	  for (long f = 0; f < nfreq; f++)
		{
      u[n_ar-1-n][f] = Su_ar[n][f];
      v[n_ar-1-n][f] = Sv_ar[n][f];
		}
  }

  for (long n = 0; n < n_r; n++)
  {
	  for (long f = 0; f < nfreq; f++)
		{
      u[n_ar+n][f] = Su_r[n][f];
      v[n_ar+n][f] = Sv_r[n][f];
		}
  }




  // SOLVE FEAUTRIER RECURSION RELATION
  // __________________________________


  // Elimination step

	for (long f = 0; f < nfreq; f++)
	{
    u[0][f] = u[0][f] / B0[f];
    v[0][f] = v[0][f] / B0[f];

    F[0][f] = B0_min_C0[f] / C[0][f];
	}

  for (long n = 1; n < ndep-1; n++)
  {
	  for (long f = 0; f < nfreq; f++)
		{
      F[n][f] = (1.0 + A[n][f]*F[n-1][f]/(1.0 + F[n-1][f])) / C[n][f];

      u[n][f] = (u[n][f] + A[n][f]*u[n-1][f]) / ((1.0 + F[n][f]) * C[n][f]);
      v[n][f] = (v[n][f] + A[n][f]*v[n-1][f]) / ((1.0 + F[n][f]) * C[n][f]);
		}
  }

	for (long f = 0; f < nfreq; f++)
	{
    u[ndep-1][f] = (u[ndep-1][f] + A[ndep-1][f]*u[ndep-2][f])
                      / (Bd_min_Ad[f] + Bd[f]*F[ndep-2][f]) * (1.0 + F[ndep-2][f]);

    v[ndep-1][f] = (v[ndep-1][f] + A[ndep-1][f]*v[ndep-2][f])
                      / (Bd_min_Ad[f] + Bd[f]*F[ndep-2][f]) * (1.0 + F[ndep-2][f]);
  
    G[ndep-1][f] = Bd_min_Ad[f] / A[ndep-1][f];
	}


  // Back substitution

  for (long n = ndep-2; n > 0; n--)
  {
	  for (long f = 0; f < nfreq; f++)
		{
      u[n][f] = u[n][f] + u[n+1][f]/(1.0+F[n][f]);
      v[n][f] = v[n][f] + v[n+1][f]/(1.0+F[n][f]);

      G[n][f] = (1.0 + C[n][f]*G[n+1][f]/(1.0+G[n+1][f])) / A[n][f];
		}
  }

  for (long f = 0; f < nfreq; f++)
	{
    u[0][f] = u[0][f] + u[1][f]/(1.0+F[0][f]);
    v[0][f] = v[0][f] + v[1][f]/(1.0+F[0][f]);
	}



  // CALCULATE LAMBDA OPERATOR
  // _________________________


  // Calculate diagonal elements

  for (long f = 0; f < nfreq; f++)
	{
    Lambda[f](0,0) = (1.0 + G[1][f]) / (B0_min_C0[f] + B0[f]*G[1][f]);
	}

  for (long n = 1; n < ndep-1; n++)
  {
    for (long f = 0; f < nfreq; f++)
		{
      Lambda[f](n,n) = (1.0 + G[n+1][f]) / ((F[n][f] + G[n+1][f] + F[n][f]*G[n+1][f]) * C[n][f]);
		}
  }

  for (long f = 0; f < nfreq; f++)
	{
    Lambda[f](ndep-1,ndep-1) = (1.0 + F[ndep-2][f]) / (Bd_min_Ad[f] + Bd[f]*F[ndep-2][f]);
	}

  // Add upper-diagonal elements

  for (long m = 1; m < ndiag; m++)
  {	  
    for (long n = 0; n < ndep-m; n++)
    {
      for (long f = 0; f < nfreq; f++)
			{
        Lambda[f](n,n+m) = Lambda[f](n+1,n+m) / (1.0 + F[n][f]);
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
        Lambda[f](n,n-m) = Lambda[f](n-1,n-m) / (1.0 + G[n][f]);
		  }
    }
  }


  return (0);

}
