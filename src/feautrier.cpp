// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "declarations.hpp"

#if (CELL_BASED)

#include "feautrier.hpp"


// feautrier: solve Feautrier recursion relation
// --------------------------------------------------

int feautrier (long ndep, long o, long r, double *S, double *dtau,
                    double *u, double *L_diag_approx)
{

  // Method described in Rybicki & Hummer (1991)

  double *A = new double[ndep];   // A coefficient in Feautrier recursion relation
  double *C = new double[ndep];   // C coefficient in Feautrier recursion relation
  double *F = new double[ndep];   // helper variable from Rybicki & Hummer (1991)
  double *G = new double[ndep];   // helper variable from Rybicki & Hummer (1991)




  // SETUP FEAUTRIER RECURSION RELATION
  // __________________________________


  // Define Feautrier A and C and B at boundary

  A[0] = 0.0;
  C[0] = 2.0/dtau[0]/dtau[0];

  double B0 = 1.0 + 2.0/dtau[0] + 2.0/dtau[0]/dtau[0];


  for (long n = 1; n < ndep-1; n++)
  {
    A[n] = 2.0 / (dtau[n] + dtau[n+1]) / dtau[n];
    C[n] = 2.0 / (dtau[n] + dtau[n+1]) / dtau[n+1];
  }


  A[ndep-1] = 2.0/dtau[ndep-1]/dtau[ndep-1];
  C[ndep-1] = 0.0;

  double Bndepm1     = 1.0 + 2.0/dtau[ndep-1] + 2.0/dtau[ndep-1]/dtau[ndep-1];

  double Bnd_min_And = 1.0 + 2.0/dtau[ndep-1];   // B[ndep-1] - A[ndep-1]

  double B0_min_C0   = 1.0 + 2.0/dtau[0];        // B[0] - C[0]


  // Store source function S initially in u

  for (long n = 0; n < ndep; n++)
  {
    u[n] = S[n];
  }




  // SOLVE FEAUTRIER RECURSION RELATION
  // __________________________________


  // Elimination step

  u[0] = u[0] / B0;

  F[0] = B0_min_C0 / C[0];


  for (long n = 1; n < ndep-1; n++)
  {
    F[n] = (1.0 + A[n]*F[n-1]/(1.0 + F[n-1])) / C[n];

    u[n] = (u[n] + A[n]*u[n-1]) / (1.0 + F[n]) / C[n];
  }


  u[ndep-1] = (u[ndep-1] + A[ndep-1]*u[ndep-2])
              / (Bnd_min_And + Bndepm1*F[ndep-2]) * (1.0 + F[ndep-2]);

  G[ndep-1] = Bnd_min_And / A[ndep-1];


  // Back substitution

  for (long n = ndep-2; n > 0; n--)
  {
    u[n] = u[n] + u[n+1]/(1.0+F[n]);

    G[n] = (1.0 + C[n]*G[n+1]/(1.0+G[n+1])) / A[n];
  }

  u[0] = u[0] + u[1]/(1.0+F[0]);




  // CALCULATE LAMBDA OPERATOR
  // _________________________


  L_diag_approx[0] = (1.0 + G[1]) / (B0_min_C0 + B0*G[1]);


  for (long n = 1; n < ndep-1; n++)
  {
    L_diag_approx[n] = (1.0 + G[n+1]) / (F[n-1] + G[n+1] + F[n-1]*G[n+1]) / C[n];
  }


  L_diag_approx[ndep-1] = (1.0 + F[ndep-2]) / (Bnd_min_And + Bndepm1*F[ndep-2]);




  // Free allocated memory for temporary variables

  delete [] A;
  delete [] C;
  delete [] F;
  delete [] G;


  return(0);

}


#endif // if CELL_BASED
