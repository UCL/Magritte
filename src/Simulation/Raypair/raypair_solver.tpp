// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


///  Feautrier solver for mean intensity (u) and flux (v) along the raypair
///////////////////////////////////////////////////////////////////////////

inline void RayPair ::
    solve_using_scattering (void)

{


  // SETUP FEAUTRIER RECURSION RELATION
  // __________________________________

  inverse_dtau0 = 1.0 / dtau[0];
  inverse_dtaud = 1.0 / dtau[ndep-2];


       C[0] =        2.0 * inverse_dtau0 * inverse_dtau0;
  B0_min_C0 = vOne + 2.0 * inverse_dtau0;

          B0 = B0_min_C0 + C[0];
  inverse_B0 = 1.0 / B0;

  Su[0] = term1[0] - (term2[1] + term2[0] - 2.0*I_bdy_0) * inverse_dtau0;
  Sv[0] = term2[0] - (term1[1] + term1[0] - 2.0*I_bdy_0) * inverse_dtau0;

  for (long n = 1; n < ndep-1; n++)
  {
    inverse_A[n] = 0.5 * (dtau[n-1] + dtau[n]) * dtau[n-1];
            A[n] = 1.0 / inverse_A[n];

    inverse_C[n] = 0.5 * (dtau[n-1] + dtau[n]) * dtau[n];
            C[n] = 1.0 / inverse_C[n];

    Su[n] = term1[n] - dtau[n] * C[n] * (term2[n+1] - term2[n-1]);
    Sv[n] = term2[n] - dtau[n] * C[n] * (term1[n+1] - term1[n-1]);
  }

  A[ndep-1] =        2.0 * inverse_dtaud * inverse_dtaud;
  Bd_min_Ad = vOne + 2.0 * inverse_dtaud;

  Bd = Bd_min_Ad + A[ndep-1];

  Su[ndep-1] = term1[ndep-1] + (term2[ndep-1] + term2[ndep-2] + 2.0*I_bdy_n) * inverse_dtaud;
  Sv[ndep-1] = term2[ndep-1] + (term1[ndep-1] + term1[ndep-2] - 2.0*I_bdy_n) * inverse_dtaud;


  // Add third order terms of the boundary condition

//  Su[0] += (term1[1] - term1[0] + (Ibdy_0 - term2[0] - C[1]*term2[2] + //(C[1]+A[1])*term2[1] - A[1]*term2[0]) * dtau[0]) / 3.0;
//  Sv[0] += (term2[1] - term2[0] + (Ibdy_0 - term1[0] - C[1]*term1[2] + //(C[1]+A[1])*term1[1] - A[1]*term1[0]) * dtau[0]) / 3.0;

//  Su[ndep-1] += (term1[ndep-1] - term1[ndep-2] + (Ibdy_n - term2[ndep-1] - C[ndep-//2]*term2[ndep-1] + (C[ndep-2]+A[ndep-2])*term2[ndep-2] - A[ndep-2]*term2[ndep-3]) * //dtau[ndep-2]) / 3.0;
//  Sv[ndep-1] += (term2[ndep-1] - term2[ndep-2] + (Ibdy_n - term1[ndep-1] - C[ndep-//2]*term1[ndep-1] + (C[ndep-2]+A[ndep-2])*term1[ndep-2] - A[ndep-2]*term1[ndep-3]) * //dtau[ndep-2]) / 3.0;


  // Add fourth order terms of the boundary condition

  //vReal T0 = 3.0 / (dtau[0]      + dtau[1]      + dtau[2]     );
  //vReal Td = 3.0 / (dtau[ndep-2] + dtau[ndep-3] + dtau[ndep-4]);




  // SOLVE FEAUTRIER RECURSION RELATION
  // __________________________________


  // Elimination step


  Su[0] = Su[0] * inverse_B0;
  Sv[0] = Sv[0] * inverse_B0;

  // F[0] = (B[0] - C[0]) / C[0];
                   F[0] = 0.5 * B0_min_C0 * dtau[0] * dtau[0];
  inverse_one_plus_F[0] = 1.0 / (vOne + F[0]);

  for (long n = 1; n < ndep-1; n++)
  {
                     F[n] = (vOne + A[n]*F[n-1]*inverse_one_plus_F[n-1]) * inverse_C[n];
    inverse_one_plus_F[n] = 1.0 / (vOne + F[n]);

    Su[n] = (Su[n] + A[n]*Su[n-1]) * inverse_one_plus_F[n] * inverse_C[n];
    Sv[n] = (Sv[n] + A[n]*Sv[n-1]) * inverse_one_plus_F[n] * inverse_C[n];
  }


  denominator = 1.0 / (Bd_min_Ad + Bd*F[ndep-2]);

  Su[ndep-1] = (Su[ndep-1] + A[ndep-1]*Su[ndep-2]) * (vOne + F[ndep-2]) * denominator;
  Sv[ndep-1] = (Sv[ndep-1] + A[ndep-1]*Sv[ndep-2]) * (vOne + F[ndep-2]) * denominator;


  if (n_off_diag == 0)
  {

    if (n_ar < ndep-1)
    {
      // BACK SUBSTITUTION
      // _________________

      // G[ndep-1] = (B[ndep-1] - A[ndep-1]) / A[ndep-1];
                      G[ndep-1] = 0.5 * Bd_min_Ad * dtau[ndep-2] * dtau[ndep-2];
      G_over_one_plus_G[ndep-1] = G[ndep-1] / (vOne + G[ndep-1]);

      for (long n = ndep-2; n > n_ar; n--)
      {
        Su[n] = Su[n] + Su[n+1] * inverse_one_plus_F[n];
        Sv[n] = Sv[n] + Sv[n+1] * inverse_one_plus_F[n];

                        G[n] = (vOne + C[n]*G_over_one_plus_G[n+1]) * inverse_A[n];
        G_over_one_plus_G[n] = G[n] / (vOne + G[n]);
      }

      Su[n_ar] = Su[n_ar] + Su[n_ar+1] * inverse_one_plus_F[n_ar];
      Sv[n_ar] = Sv[n_ar] + Sv[n_ar+1] * inverse_one_plus_F[n_ar];


      // CALCULATE LAMBDA OPERATOR (DIAGONAL)
      // ____________________________________

      L_diag[n_ar] = inverse_C[n_ar] / (F[n_ar] + G_over_one_plus_G[n_ar+1]);
    }

    else
    {
      L_diag[ndep-1] = (vOne + F[ndep-2]) / (Bd_min_Ad + Bd*F[ndep-2]);
    }

  }

  else

  {
    // BACK SUBSTITUTION
    // _________________

    // G[ndep-1] = (B[ndep-1] - A[ndep-1]) / A[ndep-1];
                     G[ndep-1] = 0.5 * Bd_min_Ad * dtau[ndep-2] * dtau[ndep-2];
    inverse_one_plus_G[ndep-1] = 1.0 / (vOne + G[ndep-1]);

    for (long n = ndep-2; n > 0; n--)
    {
      Su[n] = Su[n] + Su[n+1] * inverse_one_plus_F[n];

                       G[n] = (vOne + C[n]*G[n+1]*inverse_one_plus_G[n+1]) * inverse_A[n];
      inverse_one_plus_G[n] = 1.0 / (vOne + G[n]);
    }

    Su[0] = Su[0] + Su[1] * inverse_one_plus_F[0];



    // CALCULATE LAMBDA OPERATOR
    // _________________________

    L_diag[ndep-1] = (vOne + F[ndep-2]) / (Bd_min_Ad + Bd*F[ndep-2]);

    for (long n = ndep-2; n >= 1; n--)
    {
      L_diag[n] = inverse_C[n] / (F[n] + G[n+1] * inverse_one_plus_G[n+1]);
    }

    L_diag[0] = (vOne + G[1]) / (B0_min_C0 + B0*G[1]);

    for (long n = ndep-1; n >= 0; n--)
    {
      L_upper[0][n] = L_diag[n+1] * inverse_one_plus_F[n  ];
      L_lower[0][n] = L_diag[n  ] * inverse_one_plus_G[n+1];
    }

    for (long m = 1; m < n_off_diag; m++)
    {
      for (long n = ndep-2-m; n >= 0; n++)
      {
        L_upper[m][n] = L_upper[m-1][n+1] * inverse_one_plus_F[n    ];
        L_lower[m][n] = L_lower[m-1][n  ] * inverse_one_plus_G[n+m+1];
      }
    }

  }


}




///  Feautrier solver for mean intensity (u) and flux (v) along the raypair
///////////////////////////////////////////////////////////////////////////

/// if no scattering

inline void RayPair ::
    solve (void)

{

  //for (long n = 0; n < ndep; n++)
  //{
  //  cout <<"dtau["<<n<<"] = "<<  dtau[n] << endl;
  //  cout <<"term["<<n<<"] = "<< term1[n] << endl;
  //}
  //
  //cout <<"I_bdy_0 = " << I_bdy_0 << endl;
  //cout <<"I_bdy_n = " << I_bdy_n << endl;


  // SETUP FEAUTRIER RECURSION RELATION
  // __________________________________

  inverse_dtau0 = 1.0 / dtau[first];
  inverse_dtaud = 1.0 / dtau[last-1];


   C[first] =        2.0 * inverse_dtau0 * inverse_dtau0;
  B0_min_C0 = vOne + 2.0 * inverse_dtau0;

          B0 = B0_min_C0 + C[first];
  inverse_B0 = 1.0 / B0;

  Su[first] = term1[first] + 2.0 * I_bdy_0 * inverse_dtau0;

  for (long n = first+1; n < last; n++)
  {
    inverse_A[n] = 0.5 * (dtau[n-1] + dtau[n]) * dtau[n-1];
            A[n] = 1.0 / inverse_A[n];

    inverse_C[n] = 0.5 * (dtau[n-1] + dtau[n]) * dtau[n];
            C[n] = 1.0 / inverse_C[n];

    Su[n] = term1[n];
  }

    A[last] =        2.0 * inverse_dtaud * inverse_dtaud;
  Bd_min_Ad = vOne + 2.0 * inverse_dtaud;

  Bd = Bd_min_Ad + A[last];

  Su[last] = term1[last] + 2.0 * I_bdy_n * inverse_dtaud;


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


  // ELIMINATION STEP
  //_________________


  Su[first] = Su[first] * inverse_B0;

  // F[0] = (B[0] - C[0]) / C[0];
                   F[first] = 0.5 * B0_min_C0 * dtau[first] * dtau[first];
  inverse_one_plus_F[first] = 1.0 / (vOne + F[first]);

  for (long n = first+1; n < last; n++)
  {
                     F[n] = (vOne + A[n]*F[n-1]*inverse_one_plus_F[n-1]) * inverse_C[n];
    inverse_one_plus_F[n] = 1.0 / (vOne + F[n]);

    Su[n] = (Su[n] + A[n]*Su[n-1]) * inverse_one_plus_F[n] * inverse_C[n];
  }


  denominator = 1.0 / (Bd_min_Ad + Bd*F[last-1]);

  Su[last] = (Su[last] + A[last]*Su[last-1]) * (vOne + F[last-1]) * denominator;


  if (n_off_diag == 0)
  {

    if (n_ar < last)
    {
      // BACK SUBSTITUTION
      // _________________

      // G[ndep-1] = (B[ndep-1] - A[ndep-1]) / A[ndep-1];
                      G[last] = 0.5 * Bd_min_Ad * dtau[last-1] * dtau[last-1];
      G_over_one_plus_G[last] = G[last] / (vOne + G[last]);

      for (long n = last-1; n > n_ar; n--)
      {
        Su[n] = Su[n] + Su[n+1] * inverse_one_plus_F[n];

                        G[n] = (vOne + C[n]*G_over_one_plus_G[n+1]) * inverse_A[n];
        G_over_one_plus_G[n] = G[n] / (vOne + G[n]);
      }

      Su[n_ar] = Su[n_ar] + Su[n_ar+1] * inverse_one_plus_F[n_ar];


      // CALCULATE LAMBDA OPERATOR (DIAGONAL)
      // ____________________________________

      L_diag[n_ar] = inverse_C[n_ar] / (F[n_ar] + G_over_one_plus_G[n_ar+1]);
    }

    else
    {
      L_diag[last] = (vOne + F[last-1]) / (Bd_min_Ad + Bd*F[last-1]);
    }

  }

  else

  {
    // BACK SUBSTITUTION
    // _________________

    // G[ndep-1] = (B[ndep-1] - A[ndep-1]) / A[ndep-1];
                     G[last] = 0.5 * Bd_min_Ad * dtau[last-1] * dtau[last-1];
    inverse_one_plus_G[last] = 1.0 / (vOne + G[last]);

    for (long n = last-1; n > first; n--)
    {
      Su[n] = Su[n] + Su[n+1] * inverse_one_plus_F[n];

                       G[n] = (vOne + C[n]*G[n+1]*inverse_one_plus_G[n+1]) * inverse_A[n];
      inverse_one_plus_G[n] = 1.0 / (vOne + G[n]);
    }

    Su[first] = Su[first] + Su[first+1] * inverse_one_plus_F[first];



    // CALCULATE LAMBDA OPERATOR
    // _________________________

    L_diag[last] = (vOne + F[last-1]) / (Bd_min_Ad + Bd*F[last-1]);

    for (long n = last-1; n >= first+1; n--)
    {
      L_diag[n] = inverse_C[n] / (F[n] + G[n+1] * inverse_one_plus_G[n+1]);
    }

    L_diag[first] = (vOne + G[first+1]) / (B0_min_C0 + B0*G[first+1]);

    for (long n = last; n >= first; n--)
    {
      L_upper[0][n] = L_diag[n+1] * inverse_one_plus_F[n  ];
      L_lower[0][n] = L_diag[n  ] * inverse_one_plus_G[n+1];
    }

    for (long m = 1; m < n_off_diag; m++)
    {
      for (long n = last-1-m; n >= first; n++)
      {
        L_upper[m][n] = L_upper[m-1][n+1] * inverse_one_plus_F[n    ];
        L_lower[m][n] = L_lower[m-1][n  ] * inverse_one_plus_G[n+m+1];
      }
    }

  }



//  // Calculate diagonal elements
//
//  //// Top to Bottom
//
//  L_diag[0] = (vOne + G[1]) / (B0_min_C0 + B0*G[1]);
//
//  for (long n = 1; n < ndep-1; n++)
//  {
//   L_diag[n] = (vOne + G[n+1]) / ((F[n] + G[n+1] + F[n]*G[n+1]) * C[n]);
//  }
//
//  L_diag[ndep-1] = (vOne + F[ndep-2]) / (Bd_min_Ad + Bd*F[ndep-2]);
//
//
//  if (n_off_diag > 0)
//  {
//    for (long n = 0; n < ndep-1; n++)
//    {
//      L_upper[0][n] = L_diag[n+1] * inverse_one_plus_F[n];
//      L_lower[0][n] = L_diag[n]   * inverse_one_plus_G[n+1];
//    }
//
//    for (long m = 1; m < n_off_diag; m++)
//    {
//      for (long n = 0; n < ndep-1-m; n++)
//      {
//        L_upper[m][n] = L_upper[m-1][n+1] * inverse_one_plus_F[n];
//        L_lower[m][n] = L_lower[m-1][n]   * inverse_one_plus_G[n+m+1];
//      }
//    }
//  }


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




///  get_Im: Get I_{-} intensity exiting first cell of the raypair.
///    @return
///////////////////////////////////////////////////////////////////

inline vReal RayPair ::
    get_Im (void)

{

  // Initialize Su

  Su[0] = term1[0]; //- (term2[1] + term2[0] - 2.0*I_bdy_0) / dtau[0];

  for (long n = 1; n < ndep-1; n++)
  {
    Su[n] = term1[n];// - 2.0 * (term2[n+1] - term2[n-1]) / (dtau[n] + dtau[n-1]);
  }

  Su[ndep-1] = term1[ndep-1]; //+ (term2[ndep-1] + term2[ndep-2] + 2.0*I_bdy_n) / dtau[ndep-2];


  // Integrate (first order) transfer equation

  vReal tau = 0.0;        // optical depth
  vReal Im  = Su[n_ar];   // intensity down the ray

  for (long n = n_ar+1; n < ndep; n++)
  {
    tau += dtau[n-1];
    Im  +=   Su[n  ] * exp(-tau);
  }

  Im += I_bdy_n * exp(-tau);


  return Im;

}




///  get_Ip: Get I_{+} intensity exiting first cell of the raypair.
///////////////////////////////////////////////////////////////////

inline vReal RayPair ::
    get_Ip (void)

{

  Su[0] = term1[0];// - (term2[1] + term2[0] - 2.0*I_bdy_0) / dtau[0];

  for (long n = 1; n < ndep-1; n++)
  {
    Su[n] = term1[n];// - 2.0 * (term2[n+1] - term2[n-1]) / (dtau[n] + dtau[n-1]);
  }

  Su[ndep-1] = term1[ndep-1]; //+ (term2[ndep-1] + term2[ndep-2] + 2.0*I_bdy_n) / dtau[ndep-2];


  vReal tau = 0.0;        // optical depth
  vReal Ip  = Su[n_ar];   // intensity down the ray

  for (long n = n_ar; n > 0; n--)
  {
    tau += dtau[n-1];
    Ip  +=   Su[n-1] * exp(-tau);
    //cout << "tau = " << tau << "\t dtau = " << dtau[n-1] << "\t Su =" << Su[n-1] << "\t Ip = " << Ip << endl;
  }

  Ip += I_bdy_0 * exp(-tau);


  return Ip;

}
