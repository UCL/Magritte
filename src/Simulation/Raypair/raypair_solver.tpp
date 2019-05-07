// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


inline void RayPair ::
    solve (void)

{

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

  Su[0] = term1[0] - (term2[1] + term2[0] - 2.0*I_bdy_0) / dtau[0];
  Sv[0] = term2[0] - (term1[1] + term1[0] - 2.0*I_bdy_0) / dtau[0];

  for (long n = 1; n < ndep-1; n++)
  {
    A[n] = 2.0 / ((dtau[n-1] + dtau[n]) * dtau[n-1]);
    C[n] = 2.0 / ((dtau[n-1] + dtau[n]) * dtau[n]);

    Su[n] = term1[n] - 2.0 * (term2[n+1] - term2[n-1]) / (dtau[n] + dtau[n-1]);
    Sv[n] = term2[n] - 2.0 * (term1[n+1] - term1[n-1]) / (dtau[n] + dtau[n-1]);
  }

  A[ndep-1] = 2.0/(dtau[ndep-2]*dtau[ndep-2]);
  C[ndep-1] = 0.0;

  Bd        = vOne + 2.0/dtau[ndep-2] + 2.0/(dtau[ndep-2]*dtau[ndep-2]);
  Bd_min_Ad = vOne + 2.0/dtau[ndep-2];

  Su[ndep-1] = term1[ndep-1] + (term2[ndep-1] + term2[ndep-2] + 2.0*I_bdy_n) / dtau[ndep-2];
  Sv[ndep-1] = term2[ndep-1] + (term1[ndep-1] + term1[ndep-2] - 2.0*I_bdy_n) / dtau[ndep-2];


  // Add third order terms of the boundary condition

//  Su[0] += (term1[1] - term1[0] + (Ibdy_0 - term2[0] - C[1]*term2[2] + (C[1]+A[1])*term2[1] - A[1]*term2[0]) * dtau[0]) / 3.0;
//  Sv[0] += (term2[1] - term2[0] + (Ibdy_0 - term1[0] - C[1]*term1[2] + (C[1]+A[1])*term1[1] - A[1]*term1[0]) * dtau[0]) / 3.0;

//  Su[ndep-1] += (term1[ndep-1] - term1[ndep-2] + (Ibdy_n - term2[ndep-1] - C[ndep-2]*term2[ndep-1] + (C[ndep-2]+A[ndep-2])*term2[ndep-2] - A[ndep-2]*term2[ndep-3]) * dtau[ndep-2]) / 3.0;
//  Sv[ndep-1] += (term2[ndep-1] - term2[ndep-2] + (Ibdy_n - term1[ndep-1] - C[ndep-2]*term1[ndep-1] + (C[ndep-2]+A[ndep-2])*term1[ndep-2] - A[ndep-2]*term1[ndep-3]) * dtau[ndep-2]) / 3.0;


  // Add fourth order terms of the boundary condition

  //vReal T0 = 3.0 / (dtau[0]      + dtau[1]      + dtau[2]     );
  //vReal Td = 3.0 / (dtau[ndep-2] + dtau[ndep-3] + dtau[ndep-4]);




  // SOLVE FEAUTRIER RECURSION RELATION
  // __________________________________


  // Elimination step

  Su[0] = Su[0] / B0;
  Sv[0] = Sv[0] / B0;

  F[0] = B0_min_C0 / C[0];

  for (long n = 1; n < ndep-1; n++)
  {
    F[n] = (vOne + A[n]*F[n-1]/(vOne + F[n-1])) / C[n];

    Su[n] = (Su[n] + A[n]*Su[n-1]) / ((vOne + F[n]) * C[n]);
    Sv[n] = (Sv[n] + A[n]*Sv[n-1]) / ((vOne + F[n]) * C[n]);
  }

  Su[ndep-1] = (Su[ndep-1] + A[ndep-1]*Su[ndep-2]) * (vOne + F[ndep-2])
               / (Bd_min_Ad + Bd*F[ndep-2]);

  Sv[ndep-1] = (Sv[ndep-1] + A[ndep-1]*Sv[ndep-2]) * (vOne + F[ndep-2])
               / (Bd_min_Ad + Bd*F[ndep-2]);

  G[ndep-1] = Bd_min_Ad / A[ndep-1];


  // Back substitution

  for (long n = ndep-2; n > 0; n--)
  {
    Su[n] = Su[n] + Su[n+1] / (vOne + F[n]);
    Sv[n] = Sv[n] + Sv[n+1] / (vOne + F[n]);

    G[n] = (vOne + C[n] * G[n+1] / (vOne+G[n+1])) / A[n];
  }

  Su[0] = Su[0] + Su[1] / (vOne + F[0]);
  Sv[0] = Sv[0] + Sv[1] / (vOne + F[0]);



  // CALCULATE LAMBDA OPERATOR
  // _________________________


  // Calculate diagonal elements

  L_diag[0] = (vOne + G[1]) / (B0_min_C0 + B0*G[1]);

  for (long n = 1; n < ndep-1; n++)
  {
   L_diag[n] = (vOne + G[n+1]) / ((F[n] + G[n+1] + F[n]*G[n+1]) * C[n]);
  }

  L_diag[ndep-1] = (vOne + F[ndep-2]) / (Bd_min_Ad + Bd*F[ndep-2]);


  if (n_off_diag > 0)
  {
    for (long n = 0; n < ndep-1; n++)
    {
      L_upper[0][n] = L_diag[n+1] / (vOne + F[n]);
      L_lower[0][n] = L_diag[n]   / (vOne + G[n+1]);
    }

    for (long m = 1; m < n_off_diag; m++)
    {
      for (long n = 0; n < ndep-1-m; n++)
      {
        L_upper[m][n] = L_upper[m-1][n+1] / (vOne + F[n]);
        L_lower[m][n] = L_lower[m-1][n]   / (vOne + G[n+m+1]);
      }
    }
  }


}




//inline void RAYPAIR ::
//    solve_ndep_is_1 (void)
//
//{
//
//  // SETUP FEAUTRIER RECURSION RELATION
//  // __________________________________
//
//  vReal inverse_dtau = 1.0 / dtau[0];
//
//  vReal A0 = 2.0 * inverse_dtau * inverse_dtau;
//
//  vReal        B0 = vOne + 2.0*inverse_dtau + 2.0*inverse_dtau*inverse_dtau;
//  vReal B0_min_A0 = vOne + 2.0*inverse_dtau;
//  vReal B0_pls_A0 = vOne + 2.0*inverse_dtau + 4.0*inverse_dtau*inverse_dtau;
//
//  vReal inverse_denominator = 1.0 / (B0_min_A0 * B0_pls_A0);
//
//
//
//
//  // SOLVE FEAUTRIER RECURSION RELATION
//  // __________________________________
//
//
//  vReal u0 = (B0*Su[0] + A0*Su[1]) * inverse_denominator;
//  vReal u1 = (B0*Su[1] + A0*Su[0]) * inverse_denominator;
//
//  Su[0] = u0;
//  Su[1] = u1;
//
//  vReal v0 = (B0*Sv[0] + A0*Sv[1]) * inverse_denominator;
//  vReal v1 = (B0*Sv[1] + A0*Sv[0]) * inverse_denominator;
//
//  Sv[0] = v0;
//  Sv[1] = v1;
//
//
//  // CALCULATE LAMBDA OPERATOR
//  // _________________________
//
//
//  // Calculate diagonal elements
//
//  ////L[f](0,0) = (1.0 + G[1][f]) / (B0_min_C0[f] + B0[f]*G[1][f]);
//  //L[0] = (vOne + G[1]) / (B0_min_C0 + B0*G[1]);
//
//  //for (long n = 1; n < ndep-1; n++)
//  //{
//  //  //L[f](n,n) = (1.0 + G[n+1][f]) / ((F[n][f] + G[n+1][f] + F[n][f]*G[n+1][f]) * C[n][f]);
//  //  L[n] = (vOne + G[n+1]) / ((F[n] + G[n+1] + F[n]*G[n+1]) * C[n]);
//  //}
//
//  ////L[f](ndep-1,ndep-1) = (1.0 + F[ndep-2][f]) / (Bd_min_Ad[f] + Bd[f]*F[ndep-2][f]);
//  //L[ndep-1] = (vOne + F[ndep-2]) / (Bd_min_Ad + Bd*F[ndep-2]);
//
//
//  //// Set number of off-diagonals to add
//  //
//  //const long ndiag = 0;
//
//  //// Add upper-diagonal elements
//
//  //for (long m = 1; m < ndiag; m++)
//  //{
//  //  for (long n = 0; n < ndep-m; n++)
//  //  {
//  //    for (long f = 0; f < nfreq_red; f++)
//	//		{
//  //      L[f](n,n+m) = L[f](n+1,n+m) / (1.0 + F[n][f]);
//	//		}
//  //  }
//  //}
//
//
//  //// Add lower-diagonal elements
//
//  //for (long m = 1; m < ndiag; m++)
//  //{
//  //  for (long n = m; n < ndep; n++)
//  //  {
//  //    for (long f = 0; f < nfreq_red; f++)
//  //    {
//  //      L[f](n,n-m) = L[f](n-1,n-m) / (1.0 + G[n][f]);
//	//	  }
//  //  }
//  //}
//
//}
