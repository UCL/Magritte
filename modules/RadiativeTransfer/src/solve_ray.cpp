// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <vector>
using namespace std;
#include <Eigen/Core>
using namespace Eigen;

#include "solve_ray.hpp"

#define CF(c,f) ( nfreq*(c) + (f) )

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

int solve_ray (const long n_r,  vector<double>& Su_r,  vector<double>& Sv_r,  vector<double>& dtau_r,
	             const long n_ar, vector<double>& Su_ar, vector<double>& Sv_ar, vector<double>& dtau_ar,
	             const long ndep, const long nfreq,      vector<double>& u,     vector<double>& v,
							 const long ndiag, Ref<MatrixXd> Lambda)
{

	vector<double> B0        (nfreq);   // B[CF(0,f)]
  vector<double> B0_min_C0 (nfreq);   // B[CF(0,f)] - C[CF(0,f)]
  vector<double> Bd        (nfreq);   // B[CF(ndep-1,f)]
	vector<double> Bd_min_Ad (nfreq);   // B[CF(ndep-1,f)] - A[CF(ndep-1,f)]

  vector<double> A (ndep*nfreq);   // A coefficient in Feautrier recursion relation
	vector<double> C (ndep*nfreq);   // C coefficient in Feautrier recursion relation
  vector<double> F (ndep*nfreq);   // helper variable from Rybicki & Hummer (1991)
  vector<double> G (ndep*nfreq);   // helper variable from Rybicki & Hummer (1991)




  // SETUP FEAUTRIER RECURSION RELATION
  // __________________________________


	if ( (n_ar > 0) && (n_r > 0) )
	{
		for (long f = 0; f < nfreq; f++)
		{
      A[CF(n_ar-1,f)] = 2.0 / ((dtau_ar[CF(0,f)] + dtau_r[CF(0,f)]) * dtau_ar[CF(0,f)]);
      C[CF(n_ar-1,f)] = 2.0 / ((dtau_ar[CF(0,f)] + dtau_r[CF(0,f)]) * dtau_r[CF(0,f)]);
		}
	}


	if (n_ar > 0)
	{
		for (long f = 0; f < nfreq; f++)
		{
      A[CF(0,f)] = 0.0;
      C[CF(0,f)] = 2.0/(dtau_ar[CF(n_ar-1,f)]*dtau_ar[CF(n_ar-1,f)]);

      B0       [f] = 1.0 + 2.0/dtau_ar[CF(n_ar-1,f)] + 2.0/(dtau_ar[CF(n_ar-1,f)]*dtau_ar[CF(n_ar-1,f)]);
      B0_min_C0[f] = 1.0 + 2.0/dtau_ar[CF(n_ar-1,f)];
		}

    for (long n = n_ar-1; n > 1; n--)
    {
		  for (long f = 0; f < nfreq; f++)
			{		
        A[CF(n_ar-n,f)] = 2.0 / ((dtau_ar[CF(n-1,f)] + dtau_ar[CF(n-2,f)]) * dtau_ar[CF(n-1,f)]);
        C[CF(n_ar-n,f)] = 2.0 / ((dtau_ar[CF(n-1,f)] + dtau_ar[CF(n-2,f)]) * dtau_ar[CF(n-2,f)]);
			}
    }
	}


	if (n_r > 0)
	{
    for (long n = 0; n < n_r-1; n++)
		{
		  for (long f = 0; f < nfreq; f++)
			{
        A[CF(n_ar+n,f)] = 2.0 / ((dtau_r[CF(n,f)] + dtau_r[CF(n+1,f)]) * dtau_r[CF(n,f)]);
        C[CF(n_ar+n,f)] = 2.0 / ((dtau_r[CF(n,f)] + dtau_r[CF(n+1,f)]) * dtau_r[CF(n+1,f)]);
			}
    }

	  for (long f = 0; f < nfreq; f++)
		{
      A[CF(ndep-1,f)] = 2.0/(dtau_r[CF(n_r-1,f)]*dtau_r[CF(n_r-1,f)]);
      C[CF(ndep-1,f)] = 0.0;

      Bd       [f] = 1.0 + 2.0/dtau_r[CF(n_r-1,f)] + 2.0/(dtau_r[CF(n_r-1,f)]*dtau_r[CF(n_r-1,f)]);
      Bd_min_Ad[f] = 1.0 + 2.0/dtau_r[CF(n_r-1,f)];
		}
	}


	if (n_ar == 0)
	{
	  for (long f = 0; f < nfreq; f++)
		{
      A[CF(0,f)] = 0.0;
      C[CF(0,f)] = 2.0/(dtau_r[CF(0,f)]*dtau_r[CF(0,f)]);

      B0       [f] = 1.0 + 2.0/dtau_r[CF(0,f)] + 2.0/(dtau_r[CF(0,f)]*dtau_r[CF(0,f)]);
      B0_min_C0[f] = 1.0 + 2.0/dtau_r[CF(0,f)];
		}
	}


	if (n_r == 0)
	{
	  for (long f = 0; f < nfreq; f++)
		{
      A[CF(ndep-1,f)] = 2.0/(dtau_ar[CF(0,f)]*dtau_ar[CF(0,f)]);
      C[CF(ndep-1,f)] = 0.0;

      Bd       [f] = 1.0 + 2.0/dtau_ar[CF(0,f)] + 2.0/(dtau_ar[CF(0,f)]*dtau_ar[CF(0,f)]);
      Bd_min_Ad[f] = 1.0 + 2.0/dtau_ar[CF(0,f)];
		}
	}


  // Store source function S initially in u

  for (long n = n_ar-1; n >= 0; n--)
  {
	  for (long f = 0; f < nfreq; f++)
		{
      u[CF(n_ar-1-n,f)] = Su_ar[CF(n,f)];
      v[CF(n_ar-1-n,f)] = Sv_ar[CF(n,f)];
		}
  }

  for (long n = 0; n < n_r; n++)
  {
	  for (long f = 0; f < nfreq; f++)
		{
      u[CF(n_ar+n,f)] = Su_r[CF(n,f)];
      v[CF(n_ar+n,f)] = Sv_r[CF(n,f)];
		}
  }




  // SOLVE FEAUTRIER RECURSION RELATION
  // __________________________________


  // Elimination step

	for (long f = 0; f < nfreq; f++)
	{
    u[CF(0,f)] = u[CF(0,f)] / B0[f];
    v[CF(0,f)] = v[CF(0,f)] / B0[f];

    F[CF(0,f)] = B0_min_C0[f] / C[CF(0,f)];
	}

  for (long n = 1; n < ndep-1; n++)
  {
	  for (long f = 0; f < nfreq; f++)
		{
      F[CF(n,f)] = (1.0 + A[CF(n,f)]*F[CF(n-1,f)]/(1.0 + F[CF(n-1,f)])) / C[CF(n,f)];

      u[CF(n,f)] = (u[CF(n,f)] + A[CF(n,f)]*u[CF(n-1,f)]) / ((1.0 + F[CF(n,f)]) * C[CF(n,f)]);
      v[CF(n,f)] = (v[CF(n,f)] + A[CF(n,f)]*v[CF(n-1,f)]) / ((1.0 + F[CF(n,f)]) * C[CF(n,f)]);
		}
  }

	for (long f = 0; f < nfreq; f++)
	{
    u[CF(ndep-1,f)] = (u[CF(ndep-1,f)] + A[CF(ndep-1,f)]*u[CF(ndep-2,f)])
                      / (Bd_min_Ad[f] + Bd[f]*F[CF(ndep-2,f)]) * (1.0 + F[CF(ndep-2,f)]);

    v[CF(ndep-1,f)] = (v[CF(ndep-1,f)] + A[CF(ndep-1,f)]*v[CF(ndep-2,f)])
                      / (Bd_min_Ad[f] + Bd[f]*F[CF(ndep-2,f)]) * (1.0 + F[CF(ndep-2,f)]);
  
    G[CF(ndep-1,f)] = Bd_min_Ad[f] / A[CF(ndep-1,f)];
	}


  // Back substitution

  for (long n = ndep-2; n > 0; n--)
  {
	  for (long f = 0; f < nfreq; f++)
		{
      u[CF(n,f)] = u[CF(n,f)] + u[CF(n+1,f)]/(1.0+F[CF(n,f)]);
      v[CF(n,f)] = v[CF(n,f)] + v[CF(n+1,f)]/(1.0+F[CF(n,f)]);

      G[CF(n,f)] = (1.0 + C[CF(n,f)]*G[CF(n+1,f)]/(1.0+G[CF(n+1,f)])) / A[CF(n,f)];
		}
  }

  for (long f = 0; f < nfreq; f++)
	{
    u[CF(0,f)] = u[CF(0,f)] + u[CF(1,f)]/(1.0+F[CF(0,f)]);
    v[CF(0,f)] = v[CF(0,f)] + v[CF(1,f)]/(1.0+F[CF(0,f)]);
	}



  // CALCULATE LAMBDA OPERATOR
  // _________________________


  // Calculate diagonal elements

  Lambda(0,0) = (1.0 + G[CF(1,f)]) / (B0_min_C0[f] + B0[f]*G[CF(1,f)]);

  for (long n = 1; n < ndep-1; n++)
  {
    Lambda(n,n) = (1.0 + G[CF(n+1,f)]) / ((F[CF(n,f)] + G[CF(n+1,f)] + F[CF(n,f)]*G[CF(n+1,f)]) * C[CF(n,f)]);
  }

  Lambda(ndep-1,ndep-1) = (1.0 + F[CF(ndep-2,f)]) / (Bd_min_Ad[f] + Bd[f]*F[CF(ndep-2,f)]);


  // Add upper-diagonal elements

  for (long m = 1; m < ndiag; m++)
  {	  
    for (long n = 0; n < ndep-m; n++)
    {
      Lambda(n,n+m) = Lambda(n+1,n+m) / (1.0 + F[CF(n,f)]);
    }
  }


  // Add lower-diagonal elements

  for (long m = 1; m < ndiag; m++)
  {	  
    for (long n = m; n < ndep; n++)
    {
      Lambda(n,n-m) = Lambda(n-1,n-m) / (1.0 + G[CF(n,f)]);
    }
  }


  return (0);

}
