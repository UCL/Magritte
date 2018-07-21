// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <iostream>
using namespace std;
#include <Eigen/Core>
using namespace Eigen;

#include "solve_ray.hpp"
#include "timer.hpp"
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
///    @param[in] f: frequency index under consideration
///    @param[out] u: reference to resulting Feautrier mean intensity vector
///    @param[out] v: reference to resulting Feautrier flux intensity vector
///    @param[in] ndiag: degree of approximation in ALO (e.g. 0->diag, 1->tridiag)
///    @param[out] Lambda: approximate Lambda operator (ALO) for this ray pair
//////////////////////////////////////////////////////////////////////////////////

int solve_ray (const long n_r,  const vReal* Su_r,  const vReal* Sv_r,  const vReal* dtau_r,
	             const long n_ar, const vReal* Su_ar, const vReal* Sv_ar, const vReal* dtau_ar,
							     vReal* A,          vReal* C,           vReal* F,           vReal* G,
						  		 vReal& B0,         vReal& B0_min_C0,   vReal& Bd,          vReal& Bd_min_Ad,
	             const long ndep,       vReal* u,           vReal* v,
							 const long ndiag,      vReal* Lambda)
{


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

    B0        = vOne + 2.0/dtau_ar[n_ar-1] + 2.0/(dtau_ar[n_ar-1]*dtau_ar[n_ar-1]);
    B0_min_C0 = vOne + 2.0/dtau_ar[n_ar-1];

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

    Bd        = vOne + 2.0/dtau_r[n_r-1] + 2.0/(dtau_r[n_r-1]*dtau_r[n_r-1]);
    Bd_min_Ad = vOne + 2.0/dtau_r[n_r-1];
	}


	if (n_ar == 0)
	{
    A[0] = 0.0;
    C[0] = 2.0/(dtau_r[0]*dtau_r[0]);

    B0        = vOne + 2.0/dtau_r[0] + 2.0/(dtau_r[0]*dtau_r[0]);
    B0_min_C0 = vOne + 2.0/dtau_r[0];
	}


	if (n_r == 0)
	{
    A[ndep-1] = 2.0/(dtau_ar[0]*dtau_ar[0]);
    C[ndep-1] = 0.0;

    Bd        = vOne + 2.0/dtau_ar[0] + 2.0/(dtau_ar[0]*dtau_ar[0]);
    Bd_min_Ad = vOne + 2.0/dtau_ar[0];
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
    F[n] = (vOne + A[n]*F[n-1]/(vOne + F[n-1])) / C[n];

    u[n] = (u[n] + A[n]*u[n-1]) / ((vOne + F[n]) * C[n]);
    v[n] = (v[n] + A[n]*v[n-1]) / ((vOne + F[n]) * C[n]);
  }

  u[ndep-1] = (u[ndep-1] + A[ndep-1]*u[ndep-2])
              / (Bd_min_Ad + Bd*F[ndep-2]) * (vOne + F[ndep-2]);

  v[ndep-1] = (v[ndep-1] + A[ndep-1]*v[ndep-2])
              / (Bd_min_Ad + Bd*F[ndep-2]) * (vOne + F[ndep-2]);

  G[ndep-1] = Bd_min_Ad / A[ndep-1];


  // Back substitution

  for (long n = ndep-2; n > 0; n--)
  {
    u[n] = u[n] + u[n+1]/(vOne+F[n]);
    v[n] = v[n] + v[n+1]/(vOne+F[n]);

    G[n] = (vOne + C[n]*G[n+1]/(vOne+G[n+1])) / A[n];
  }

  u[0] = u[0] + u[1]/(vOne+F[0]);
  v[0] = v[0] + v[1]/(vOne+F[0]);



  // CALCULATE LAMBDA OPERATOR
  // _________________________


  // Calculate diagonal elements

  //Lambda[f](0,0) = (1.0 + G[1][f]) / (B0_min_C0[f] + B0[f]*G[1][f]);
  Lambda[0] = (vOne + G[1]) / (B0_min_C0 + B0*G[1]);

  for (long n = 1; n < ndep-1; n++)
  {
    //Lambda[f](n,n) = (1.0 + G[n+1][f]) / ((F[n][f] + G[n+1][f] + F[n][f]*G[n+1][f]) * C[n][f]);
    Lambda[n] = (vOne + G[n+1]) / ((F[n] + G[n+1] + F[n]*G[n+1]) * C[n]);
  }

  //Lambda[f](ndep-1,ndep-1) = (1.0 + F[ndep-2][f]) / (Bd_min_Ad[f] + Bd[f]*F[ndep-2][f]);
  Lambda[ndep-1] = (vOne + F[ndep-2]) / (Bd_min_Ad + Bd*F[ndep-2]);


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
