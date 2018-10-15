// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <Eigen/Core>
using namespace Eigen;

#include "solve_ray.hpp"
#include "GridTypes.hpp"


///  solve_ray: solve radiative transfer equation using the Feautrier method
///  and the numerical scheme devised by Rybicki & Hummer (1991)
///    @param[in] ndep: number of points on ray this ray pair
///    @param[in/out] Su: in pointer to source function for u / out solution for u
///    @param[in/out] Sv: in pointer to source function for v / out solution for v
///    @param[in] dtau: pointer to optical depth increments along ray r
///    @param[in] ndep: total number of points on the combined ray r and ar
///    @param[in] ndiag: degree of approximation in ALO (e.g. 0->diag, 1->tridiag)
///    @param[out] Lambda: approximate Lambda operator (ALO) for this ray pair
//////////////////////////////////////////////////////////////////////////////////

inline int solve_ray (const long   ndep,
                            vReal *Su,
                            vReal *Sv,
                      const vReal *dtau,
                      const long   ndiag,
                            vReal *Lambda,
                      const long ncells   )
{

  vReal A[ncells];   // A coefficient in Feautrier recursion relation
  vReal C[ncells];   // C coefficient in Feautrier recursion relation
  vReal F[ncells];   // helper variable from Rybicki & Hummer (1991)
  vReal G[ncells];   // helper variable from Rybicki & Hummer (1991)

  vReal B0;          // B[0]
  vReal B0_min_C0;   // B[0] - C[0]
  vReal Bd;          // B[ndep-1]
  vReal Bd_min_Ad;   // B[ndep-1] - A[ndep-1]




  // SETUP FEAUTRIER RECURSION RELATION
  // __________________________________


  A[0] = 0.0;
  C[0] = 2.0/(dtau[0]*dtau[0]);

  B0        = vOne + 2.0/dtau[0] + 2.0/(dtau[0]*dtau[0]);
  B0_min_C0 = vOne + 2.0/dtau[0];

  for (long n = 1; n < ndep-1; n++)
  {
    A[n] = 2.0 / ((dtau[n] + dtau[n+1]) * dtau[n]);
    C[n] = 2.0 / ((dtau[n] + dtau[n+1]) * dtau[n+1]);
  }

  A[ndep-1] = 2.0/(dtau[ndep-1]*dtau[ndep-1]);
  C[ndep-1] = 0.0;

  Bd        = vOne + 2.0/dtau[ndep-1] + 2.0/(dtau[ndep-1]*dtau[ndep-1]);
  Bd_min_Ad = vOne + 2.0/dtau[ndep-1];




  // SOLVE FEAUTRIER RECURSION RELATION
  // __________________________________


  // Elimination step

  Su[0] = Su[0] / B0;
//  Sv[0] = Sv[0] / B0;

  F[0] = B0_min_C0 / C[0];

  for (long n = 1; n < ndep-1; n++)
  {
    F[n] = (vOne + A[n]*F[n-1]/(vOne + F[n-1])) / C[n];

    Su[n] = (Su[n] + A[n]*Su[n-1]) / ((vOne + F[n]) * C[n]);
//    Sv[n] = (Sv[n] + A[n]*Sv[n-1]) / ((vOne + F[n]) * C[n]);
  }

  Su[ndep-1] = (Su[ndep-1] + A[ndep-1]*Su[ndep-2]) * (vOne + F[ndep-2])
               / (Bd_min_Ad + Bd*F[ndep-2]);

//  Sv[ndep-1] = (Sv[ndep-1] + A[ndep-1]*Sv[ndep-2])
//               / (Bd_min_Ad + Bd*F[ndep-2]) * (vOne + F[ndep-2]);

//  G[ndep-1] = Bd_min_Ad / A[ndep-1];


  // Back substitution

  for (long n = ndep-2; n > 0; n--)
  {
    Su[n] = Su[n] + Su[n+1] / (vOne + F[n]);
//    Sv[n] = Sv[n] + Sv[n+1] / (vOne + F[n]);

  //  G[n] = (vOne + C[n]*G[n+1]/(vOne+G[n+1])) / A[n];
  }

  Su[0] = Su[0] + Su[1] / (vOne + F[0]);
//  Sv[0] = Sv[0] + Sv[1]/(vOne+F[0]);



  // CALCULATE LAMBDA OPERATOR
  // _________________________


  // Calculate diagonal elements

  ////Lambda[f](0,0) = (1.0 + G[1][f]) / (B0_min_C0[f] + B0[f]*G[1][f]);
  //Lambda[0] = (vOne + G[1]) / (B0_min_C0 + B0*G[1]);

  //for (long n = 1; n < ndep-1; n++)
  //{
  //  //Lambda[f](n,n) = (1.0 + G[n+1][f]) / ((F[n][f] + G[n+1][f] + F[n][f]*G[n+1][f]) * C[n][f]);
  //  Lambda[n] = (vOne + G[n+1]) / ((F[n] + G[n+1] + F[n]*G[n+1]) * C[n]);
  //}

  ////Lambda[f](ndep-1,ndep-1) = (1.0 + F[ndep-2][f]) / (Bd_min_Ad[f] + Bd[f]*F[ndep-2][f]);
  //Lambda[ndep-1] = (vOne + F[ndep-2]) / (Bd_min_Ad + Bd*F[ndep-2]);


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
