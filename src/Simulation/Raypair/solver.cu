#define I(d,f) (f + d*nfreqs)


// __global__ void solverKernel (
//     const long    nfreqs,
//     const long    n_ar,
//     const long    first,
//     const long    last,
//     const double *I_bdy_0,
//     const double *I_bdy_n,
//     const double *term1,
//     const double *term2,
//     const double *dtau,
//           double *inverse_dtau0,
//           double *inverse_dtaud,
//           double *A,
//           double *C,
//           double *F,
//           double *G,
//           double *inverse_A,
//           double *inverse_C,
//           double *inverse_one_plus_F,
//           double *inverse_one_plus_G,
//           double * G_over_one_plus_G,
//           double *B0_min_C0,
//           double *Bd_min_Ad,
//           double *B0,
//           double *Bd,
//           double *inverse_B0,
//           double *denominator,
//           double *L_diag,
//           double *Su,
//           double *Sv,
//     const long    f                  )
// {
//   /// SETUP FEAUTRIER RECURSION RELATION
//   //////////////////////////////////////
//
//   inverse_dtau0[f] = 1.0 / dtau[I(first, f)];
//   inverse_dtaud[f] = 1.0 / dtau[I(last-1,f)];
//
//   C[I(first,f)] =       2.0 * inverse_dtau0[f] * inverse_dtau0[f];
//   B0_min_C0[f]  = 1.0 + 2.0 * inverse_dtau0[f];
//
//           B0[f] = B0_min_C0[f] + C[I(first,f)];
//   inverse_B0[f] = 1.0 / B0[f];
//
//   Su[I(first,f)] = term1[I(first,f)] + 2.0 * I_bdy_0[f] * inverse_dtau0[f];
//
//   for (long n = first+1; n < last; n++)
//   {
//     inverse_A[I(n,f)] = 0.5 * (dtau[I(n-1,f)] + dtau[I(n,f)]) * dtau[I(n-1,f)];
//     inverse_C[I(n,f)] = 0.5 * (dtau[I(n-1,f)] + dtau[I(n,f)]) * dtau[I(n,  f)];
//
//     A[I(n,f)] = 1.0 / inverse_A[I(n,f)];
//     C[I(n,f)] = 1.0 / inverse_C[I(n,f)];
//
//     Su[I(n,f)] = term1[I(n,f)];
//   }
//
//   A[I(last,f)] =       2.0 * inverse_dtaud[f] * inverse_dtaud[f];
//   Bd_min_Ad[f] = 1.0 + 2.0 * inverse_dtaud[f];
//
//   Bd[f] = Bd_min_Ad[f] + A[I(last,f)];
//
//   Su[I(last,f)] = term1[I(last,f)] + 2.0 * I_bdy_n[f] * inverse_dtaud[f];
//
//
//   /// SOLVE FEAUTRIER RECURSION RELATION
//   //////////////////////////////////////
//
//
//   /// ELIMINATION STEP
//   ////////////////////
//
//
//   Su[I(first,f)] = Su[I(first,f)] * inverse_B0[f];
//
//   // F[0] = (B[0] - C[0]) / C[0];
//                    F[I(first,f)] = 0.5 * B0_min_C0[f] * dtau[I(first,f)] * dtau[I(first,f)];
//   inverse_one_plus_F[I(first,f)] = 1.0 / (1.0 + F[I(first,f)]);
//
//   for (long n = first+1; n < last; n++)
//   {
//                      F[I(n,f)] = (1.0 + A[I(n,f)]*F[I(n-1,f)]*inverse_one_plus_F[I(n-1,f)]) * inverse_C[I(n,f)];
//     inverse_one_plus_F[I(n,f)] = 1.0 / (1.0 + F[I(n,f)]);
//
//     Su[I(n,f)] = (Su[I(n,f)] + A[I(n,f)]*Su[I(n-1,f)]) * inverse_one_plus_F[I(n,f)] * inverse_C[I(n,f)];
//   }
//
//   denominator[f] = 1.0 / (Bd_min_Ad[f] + Bd[f]*F[I(last-1,f)]);
//
//   Su[I(last,f)] = (Su[I(last,f)] + A[I(last,f)]*Su[I(last-1,f)]) * (1.0 + F[I(last-1,f)]) * denominator[f];
//
//
//   if (n_ar < last)
//   {
//     /// BACK SUBSTITUTION
//     /////////////////////
//
//     // G[ndep-1] = (B[ndep-1] - A[ndep-1]) / A[ndep-1];
//                     G[I(last,f)] = 0.5 * Bd_min_Ad[f] * dtau[I(last-1,f)] * dtau[I(last-1,f)];
//     G_over_one_plus_G[I(last,f)] = G[I(last,f)] / (1.0 + G[I(last,f)]);
//
//     for (long n = last-1; n > n_ar; n--)
//     {
//       Su[I(n,f)] = Su[I(n,f)] + Su[I(n+1,f)] * inverse_one_plus_F[I(n,f)];
//
//                       G[I(n,f)] = (1.0 + C[I(n,f)]*G_over_one_plus_G[I(n+1,f)]) * inverse_A[I(n,f)];
//       G_over_one_plus_G[I(n,f)] = G[I(n,f)] / (1.0 + G[I(n,f)]);
//     }
//
//     Su[I(n_ar,f)] = Su[I(n_ar,f)] + Su[I(n_ar+1,f)] * inverse_one_plus_F[I(n_ar,f)];
//
//
//     /// CALCULATE LAMBDA OPERATOR (DIAGONAL)
//     ////////////////////////////////////////
//
//     L_diag[I(n_ar,f)] = inverse_C[I(n_ar,f)] / (F[I(n_ar,f)] + G_over_one_plus_G[I(n_ar+1,f)]);
//   }
//
//   else
//   {
//     /// CALCULATE LAMBDA OPERATOR (DIAGONAL)
//     ////////////////////////////////////////
//
//     L_diag[I(last,f)] = (1.0 + F[I(last-1,f)]) / (Bd_min_Ad[f] + Bd[f]*F[I(last-1,f)]);
//   }
//
//
//
// }


__device__
inline void gpuRayPair :: solve (const long f)
{
  /// SETUP FEAUTRIER RECURSION RELATION
  //////////////////////////////////////

  inverse_dtau0[f] = 1.0 / dtau[I(first, f)];
  inverse_dtaud[f] = 1.0 / dtau[I(last-1,f)];

  C[I(first,f)] =       2.0 * inverse_dtau0[f] * inverse_dtau0[f];
  B0_min_C0[f]  = 1.0 + 2.0 * inverse_dtau0[f];

          B0[f] = B0_min_C0[f] + C[I(first,f)];
  inverse_B0[f] = 1.0 / B0[f];

  Su[I(first,f)] = term1[I(first,f)] + 2.0 * I_bdy_0[f] * inverse_dtau0[f];

  for (long n = first+1; n < last; n++)
  {
    inverse_A[I(n,f)] = 0.5 * (dtau[I(n-1,f)] + dtau[I(n,f)]) * dtau[I(n-1,f)];
    inverse_C[I(n,f)] = 0.5 * (dtau[I(n-1,f)] + dtau[I(n,f)]) * dtau[I(n,  f)];

    A[I(n,f)] = 1.0 / inverse_A[I(n,f)];
    C[I(n,f)] = 1.0 / inverse_C[I(n,f)];

    Su[I(n,f)] = term1[I(n,f)];
  }

  A[I(last,f)] =       2.0 * inverse_dtaud[f] * inverse_dtaud[f];
  Bd_min_Ad[f] = 1.0 + 2.0 * inverse_dtaud[f];

  Bd[f] = Bd_min_Ad[f] + A[I(last,f)];

  Su[I(last,f)] = term1[I(last,f)] + 2.0 * I_bdy_n[f] * inverse_dtaud[f];


  /// SOLVE FEAUTRIER RECURSION RELATION
  //////////////////////////////////////


  /// ELIMINATION STEP
  ////////////////////


  Su[I(first,f)] = Su[I(first,f)] * inverse_B0[f];

  // F[0] = (B[0] - C[0]) / C[0];
                   F[I(first,f)] = 0.5 * B0_min_C0[f] * dtau[I(first,f)] * dtau[I(first,f)];
  inverse_one_plus_F[I(first,f)] = 1.0 / (1.0 + F[I(first,f)]);

  for (long n = first+1; n < last; n++)
  {
                     F[I(n,f)] = (1.0 + A[I(n,f)]*F[I(n-1,f)]*inverse_one_plus_F[I(n-1,f)]) * inverse_C[I(n,f)];
    inverse_one_plus_F[I(n,f)] = 1.0 / (1.0 + F[I(n,f)]);

    Su[I(n,f)] = (Su[I(n,f)] + A[I(n,f)]*Su[I(n-1,f)]) * inverse_one_plus_F[I(n,f)] * inverse_C[I(n,f)];
  }

  denominator[f] = 1.0 / (Bd_min_Ad[f] + Bd[f]*F[I(last-1,f)]);

  Su[I(last,f)] = (Su[I(last,f)] + A[I(last,f)]*Su[I(last-1,f)]) * (1.0 + F[I(last-1,f)]) * denominator[f];


  if (n_ar < last)
  {
    /// BACK SUBSTITUTION
    /////////////////////

    // G[ndep-1] = (B[ndep-1] - A[ndep-1]) / A[ndep-1];
                    G[I(last,f)] = 0.5 * Bd_min_Ad[f] * dtau[I(last-1,f)] * dtau[I(last-1,f)];
    G_over_one_plus_G[I(last,f)] = G[I(last,f)] / (1.0 + G[I(last,f)]);

    for (long n = last-1; n > n_ar; n--)
    {
      Su[I(n,f)] = Su[I(n,f)] + Su[I(n+1,f)] * inverse_one_plus_F[I(n,f)];

                      G[I(n,f)] = (1.0 + C[I(n,f)]*G_over_one_plus_G[I(n+1,f)]) * inverse_A[I(n,f)];
      G_over_one_plus_G[I(n,f)] = G[I(n,f)] / (1.0 + G[I(n,f)]);
    }

    Su[I(n_ar,f)] = Su[I(n_ar,f)] + Su[I(n_ar+1,f)] * inverse_one_plus_F[I(n_ar,f)];


    /// CALCULATE LAMBDA OPERATOR (DIAGONAL)
    ////////////////////////////////////////

    L_diag[I(n_ar,f)] = inverse_C[I(n_ar,f)] / (F[I(n_ar,f)] + G_over_one_plus_G[I(n_ar+1,f)]);
  }

  else
  {
    /// CALCULATE LAMBDA OPERATOR (DIAGONAL)
    ////////////////////////////////////////

    L_diag[I(last,f)] = (1.0 + F[I(last-1,f)]) / (Bd_min_Ad[f] + Bd[f]*F[I(last-1,f)]);
  }


  return;
}
