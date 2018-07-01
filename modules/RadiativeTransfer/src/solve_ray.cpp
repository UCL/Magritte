// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <vector>
#include <iostream>
using namespace std;
#include <Eigen/Core>
using namespace Eigen;

#include "solve_ray.hpp"
#include "types.hpp"
#include "GridTypes.hpp"


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
///    @param[in] nfreq_red: reduced number of frequency bins
///    @param[out] u: reference to resulting Feautrier mean intensity vector
///    @param[out] v: reference to resulting Feautrier flux intensity vector
///    @param[in] ndiag: degree of approximation in ALO (e.g. 0->diag, 1->tridiag)
///    @param[out] Lambda: approximate Lambda operator (ALO) for this ray pair
//////////////////////////////////////////////////////////////////////////////////

int solve_ray (const long n_r,  const vReal2& Su_r,  const vReal2& Sv_r,  const vReal2& dtau_r,
	             const long n_ar, const vReal2& Su_ar, const vReal2& Sv_ar, const vReal2& dtau_ar,
	             const long ndep, const long nfreq_red,        vReal2& u,           vReal2& v,
							 const long ndiag, vReal2& Lambda)
{

	vReal1 B0        (nfreq_red);           // B[0][f]
  vReal1 B0_min_C0 (nfreq_red);           // B[0][f] - C[0][f]
  vReal1 Bd        (nfreq_red);           // B[ndep-1][f]
	vReal1 Bd_min_Ad (nfreq_red);           // B[ndep-1][f] - A[ndep-1][f]

  vReal2 A (ndep, vReal1 (nfreq_red));   // A coefficient in Feautrier recursion relation
	vReal2 C (ndep, vReal1 (nfreq_red));   // C coefficient in Feautrier recursion relation
  vReal2 F (ndep, vReal1 (nfreq_red));   // helper variable from Rybicki & Hummer (1991)
  vReal2 G (ndep, vReal1 (nfreq_red));   // helper variable from Rybicki & Hummer (1991)




  // SETUP FEAUTRIER RECURSION RELATION
  // __________________________________


	if ( (n_ar > 0) && (n_r > 0) )
	{
		for (long f = 0; f < nfreq_red; f++)
		{
      A[n_ar-1][f] = 2.0 / ((dtau_ar[0][f] + dtau_r[0][f]) * dtau_ar[0][f]);
      C[n_ar-1][f] = 2.0 / ((dtau_ar[0][f] + dtau_r[0][f]) * dtau_r[0][f]);
		}
	}


	if (n_ar > 0)
	{
		for (long f = 0; f < nfreq_red; f++)
		{
      A[0][f] = 0.0;
      C[0][f] = 2.0/(dtau_ar[n_ar-1][f]*dtau_ar[n_ar-1][f]);

      B0       [f] = vOne + 2.0/dtau_ar[n_ar-1][f] + 2.0/(dtau_ar[n_ar-1][f]*dtau_ar[n_ar-1][f]);
      B0_min_C0[f] = vOne + 2.0/dtau_ar[n_ar-1][f];
		}

    for (long n = n_ar-1; n > 1; n--)
    {
		  for (long f = 0; f < nfreq_red; f++)
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
		  for (long f = 0; f < nfreq_red; f++)
			{
        A[n_ar+n][f] = 2.0 / ((dtau_r[n][f] + dtau_r[n+1][f]) * dtau_r[n][f]);
        C[n_ar+n][f] = 2.0 / ((dtau_r[n][f] + dtau_r[n+1][f]) * dtau_r[n+1][f]);
			}
    }

	  for (long f = 0; f < nfreq_red; f++)
		{
      A[ndep-1][f] = 2.0/(dtau_r[n_r-1][f]*dtau_r[n_r-1][f]);
      C[ndep-1][f] = 0.0;

      Bd       [f] = vOne + 2.0/dtau_r[n_r-1][f] + 2.0/(dtau_r[n_r-1][f]*dtau_r[n_r-1][f]);
      Bd_min_Ad[f] = vOne + 2.0/dtau_r[n_r-1][f];
		}
	}


	if (n_ar == 0)
	{
	  for (long f = 0; f < nfreq_red; f++)
		{
      A[0][f] = 0.0;
      C[0][f] = 2.0/(dtau_r[0][f]*dtau_r[0][f]);

      B0       [f] = vOne + 2.0/dtau_r[0][f] + 2.0/(dtau_r[0][f]*dtau_r[0][f]);
      B0_min_C0[f] = vOne + 2.0/dtau_r[0][f];
		}
	}


	if (n_r == 0)
	{
	  for (long f = 0; f < nfreq_red; f++)
		{
      A[ndep-1][f] = 2.0/(dtau_ar[0][f]*dtau_ar[0][f]);
      C[ndep-1][f] = 0.0;

      Bd       [f] = vOne + 2.0/dtau_ar[0][f] + 2.0/(dtau_ar[0][f]*dtau_ar[0][f]);
      Bd_min_Ad[f] = vOne + 2.0/dtau_ar[0][f];
		}
	}


  // Store source function S initially in u

  for (long n = n_ar-1; n >= 0; n--)
  {
	  for (long f = 0; f < nfreq_red; f++)
		{
      u[n_ar-1-n][f] = Su_ar[n][f];
      v[n_ar-1-n][f] = Sv_ar[n][f];
		}
  }

  for (long n = 0; n < n_r; n++)
  {
	  for (long f = 0; f < nfreq_red; f++)
		{
      u[n_ar+n][f] = Su_r[n][f];
      v[n_ar+n][f] = Sv_r[n][f];
		}
  }




  // SOLVE FEAUTRIER RECURSION RELATION
  // __________________________________


  // Elimination step

	for (long f = 0; f < nfreq_red; f++)
	{
    u[0][f] = u[0][f] / B0[f];
    v[0][f] = v[0][f] / B0[f];

    F[0][f] = B0_min_C0[f] / C[0][f];
	}

  for (long n = 1; n < ndep-1; n++)
  {
	  for (long f = 0; f < nfreq_red; f++)
		{
      F[n][f] = (vOne + A[n][f]*F[n-1][f]/(vOne + F[n-1][f])) / C[n][f];

      u[n][f] = (u[n][f] + A[n][f]*u[n-1][f]) / ((vOne + F[n][f]) * C[n][f]);
      v[n][f] = (v[n][f] + A[n][f]*v[n-1][f]) / ((vOne + F[n][f]) * C[n][f]);
		}
  }

	for (long f = 0; f < nfreq_red; f++)
	{
    u[ndep-1][f] = (u[ndep-1][f] + A[ndep-1][f]*u[ndep-2][f])
                      / (Bd_min_Ad[f] + Bd[f]*F[ndep-2][f]) * (vOne + F[ndep-2][f]);

    v[ndep-1][f] = (v[ndep-1][f] + A[ndep-1][f]*v[ndep-2][f])
                      / (Bd_min_Ad[f] + Bd[f]*F[ndep-2][f]) * (vOne + F[ndep-2][f]);
  
    G[ndep-1][f] = Bd_min_Ad[f] / A[ndep-1][f];
	}


  // Back substitution

  for (long n = ndep-2; n > 0; n--)
  {
	  for (long f = 0; f < nfreq_red; f++)
		{
      u[n][f] = u[n][f] + u[n+1][f]/(vOne+F[n][f]);
      v[n][f] = v[n][f] + v[n+1][f]/(vOne+F[n][f]);

      G[n][f] = (vOne + C[n][f]*G[n+1][f]/(vOne+G[n+1][f])) / A[n][f];
		}
  }

  for (long f = 0; f < nfreq_red; f++)
	{
    u[0][f] = u[0][f] + u[1][f]/(vOne+F[0][f]);
    v[0][f] = v[0][f] + v[1][f]/(vOne+F[0][f]);
	}



  // CALCULATE LAMBDA OPERATOR
  // _________________________


  // Calculate diagonal elements

  for (long f = 0; f < nfreq_red; f++)
	{
    //Lambda[f](0,0) = (1.0 + G[1][f]) / (B0_min_C0[f] + B0[f]*G[1][f]);
    Lambda[0][f] = (vOne + G[1][f]) / (B0_min_C0[f] + B0[f]*G[1][f]);
	}

  for (long n = 1; n < ndep-1; n++)
  {
    for (long f = 0; f < nfreq_red; f++)
		{
      //Lambda[f](n,n) = (1.0 + G[n+1][f]) / ((F[n][f] + G[n+1][f] + F[n][f]*G[n+1][f]) * C[n][f]);
      Lambda[n][f] = (vOne + G[n+1][f]) / ((F[n][f] + G[n+1][f] + F[n][f]*G[n+1][f]) * C[n][f]);
		}
  }

  for (long f = 0; f < nfreq_red; f++)
	{
    //Lambda[f](ndep-1,ndep-1) = (1.0 + F[ndep-2][f]) / (Bd_min_Ad[f] + Bd[f]*F[ndep-2][f]);
    Lambda[ndep-1][f] = (vOne + F[ndep-2][f]) / (Bd_min_Ad[f] + Bd[f]*F[ndep-2][f]);
	}

	
  //// Add upper-diagonal elements

  //for (long m = 1; m < ndiag; m++)
  //{	  
  //  for (long n = 0; n < ndep-m; n++)
  //  {
  //    for (long f = 0; f < nfreq_red; f++)
	//		{
  //      Lambda[f](n,n+m) = Lambda[f](n+1,n+m) / (1.0 + F[n][f]);
	//		}
  //  }
  //}


  //// Add lower-diagonal elements

  //for (long m = 1; m < ndiag; m++)
  //{	  
  //  for (long n = m; n < ndep; n++)
  //  {
  //    for (long f = 0; f < nfreq_red; f++)
  //    {    
  //      Lambda[f](n,n-m) = Lambda[f](n-1,n-m) / (1.0 + G[n][f]);
	//	  }
  //  }
  //}


  return (0);

}
