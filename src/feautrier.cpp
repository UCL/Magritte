/* Frederik De Ceuster - University College London & KU Leuven                                   */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/* Solver for the Radiative Transfer equation in 1D (i.e. along one ray) for a certain frequency */
/*                                                                                               */
/* (based on code by Dr. Jeremy Yates)                                                           */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/*     ndep          = number of depth points (evaluation points) along the ray   (IN)           */
/*     P             = (Iup + Idown)/2 = Feautrier's mean intensity              (OUT)           */
/*     A, B and C    = coefficients in the Feautrier recursion relation          ( - )           */
/*     Fd1, Fd2, Fd3 = tridiagonal elements of the Feautrier matrix              ( - )           */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/



#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "../parameters.hpp"
#include "Magritte_config.hpp"
#include "declarations.hpp"

#include "feautrier.hpp"



/* feautrier: fill Feautrier matrix and solve it                                                 */
/*-----------------------------------------------------------------------------------------------*/

int feautrier( EVALPOINT *evalpoint, long *key, long *raytot, long *cum_raytot, long gridp, long r,
               double *S, double *dtau, double *u, double *L_diag_approx )
{


  long ar = antipod[r];


#ifdef ON_THE_FLY

  long etot1 = raytot[ar];                     /* total number of evaluation points along ray ar */
  long etot2 = raytot[r];                       /* total number of evaluation points along ray r */

#else

  long etot1 = raytot[RINDEX(gridp, ar)];      /* total number of evaluation points along ray ar */
  long etot2 = raytot[RINDEX(gridp, r)];        /* total number of evaluation points along ray r */

#endif


  long ndep = etot1 + etot2;               /* nr. of depth points along a pair of antipodal rays */


  double *A;                                    /* A coefficient in Feautrier recursion relation */
  A = (double*) malloc( ndep*sizeof(double) );

  double *C;                                    /* C coefficient in Feautrier recursion relation */
  C = (double*) malloc( ndep*sizeof(double) );

  double *F;                                                                  /* helper variable */
  F = (double*) malloc( ndep*sizeof(double) );

  double *G;                                                                  /* helper variable */
  G = (double*) malloc( ndep*sizeof(double) );




  /*   SETUP THE FEAUTRIER RECURSION RELATION                                                    */
  /*_____________________________________________________________________________________________*/


  /* Define Feautrier A and C and B at the boundaries */

  A[0] = 0.0;
  C[0] = 2.0/dtau[0]/dtau[0];

  double B0 = 1.0 + 2.0/dtau[0] + 2.0/dtau[0]/dtau[0];


  for (long n=1; n<ndep-1; n++) {

    A[n] = 2.0 / (dtau[n] + dtau[n+1]) / dtau[n];
    C[n] = 2.0 / (dtau[n] + dtau[n+1]) / dtau[n+1];
  }


  A[ndep-1] = 2.0/dtau[ndep-1]/dtau[ndep-1];
  C[ndep-1] = 0.0;

  double Bndepm1 = 1.0 + 2.0/dtau[ndep-1] + 2.0/dtau[ndep-1]/dtau[ndep-1];

  double Bnd_min_And = 1.0 + 2.0/dtau[ndep-1];                          /* B[ndep-1] - A[ndep-1] */

  double B0_min_C0   = 1.0 + 2.0/dtau[0];                                         /* B[0] - C[0] */

  /* Store the source function S initially in u */

  for (long n=0; n<ndep; n++){

    u[n] = S[n];
  }

  // printf("u = %lE\n", u[etot1]);
  // printf("B0 = %lE\n", B0);
  // printf("dtau = %lE\n", dtau[0]);

  /*_____________________________________________________________________________________________*/





  /*   SOLVE THE FEAUTRIER RECURSION RELATION                                                    */
  /*_____________________________________________________________________________________________*/


  /* Elimination step */

  u[0] = u[0] / B0;

  F[0] = B0_min_C0 / C[0];

  // printf("u0 = %lE\n", u[0]);

  for (long n=1; n<ndep-1; n++){

    F[n] = ( 1.0 + A[n]*F[n-1]/(1.0 + F[n-1]) ) / C[n];

    u[n] = (u[n] + A[n]*u[n-1]) / (1.0 + F[n]) / C[n];
  }

  // printf("u1 = %lE\n", u[1]);

  u[ndep-1] = (u[ndep-1] + A[ndep-1]*u[ndep-2])
              / (Bnd_min_And + Bndepm1*F[ndep-2]) * (1.0 + F[ndep-2]);

  G[ndep-1] = Bnd_min_And / A[ndep-1];

  // printf("u1 = %lE\n", u[ndep-2]);

  /* Back substitution */

  for (long n=ndep-2; n>0; n--){

    u[n] = u[n] + u[n+1]/(1.0+F[n]);

    G[n] = (1.0 + C[n]*G[n+1]/(1.0+G[n+1])) / A[n];
  }


  u[0] = u[0] + u[1]/(1.0+F[0]);

  // printf("u1 = %lE\n", u[1]);

  /*_____________________________________________________________________________________________*/





  /*   CALCULATE THE LAMBDA OPERATOR                                                             */
  /*_____________________________________________________________________________________________*/


  L_diag_approx[0] = (1.0 + G[1]) / (B0_min_C0 + B0*G[1]);


  for (long n=1; n<ndep-1; n++){

    L_diag_approx[n] = (1.0 + G[n+1]) / (F[n-1] + G[n+1] + F[n-1]*G[n+1]) / C[n];
  }


  L_diag_approx[ndep-1] = (1.0 + F[ndep-2]) / (Bnd_min_And + Bndepm1*F[ndep-2]);

  // printf("u 0 = %lE\n", u[0]);
  // printf("L_diag_approx 0 = %lE\n", L_diag_approx[0]);
  //
  // printf("u = %lE\n", u[etot1]);
  // printf("L_diag_approx = %lE\n", L_diag_approx[etot1]);


  /*_____________________________________________________________________________________________*/





  /*   RELATE THE RESULTS BACK TO THE EVALUATION POINTS VIA                                      */
  /*_____________________________________________________________________________________________*/


//   {
//
//
// #   ifdef ON_THE_FLY
//
//     long g_p = LOCAL_GP_NR_OF_EVALP(ar,etot1-1);   /* grid point nr. of the considered evalpoint */
//     long e_p = g_p;
//
// #   else
//
//     long g_p = GP_NR_OF_EVALP(gridp,ar,etot1-1);   /* grid point nr. of the considered evalpoint */
//     long e_p = GINDEX(gridp,g_p);
//
// #   endif
//
//
//     if ( evalpoint[e_p].eqp == gridp ){
//
//       u_intensity[RINDEX(g_p,ar)] = P[0];
//     }
//   }
//
//
//   for (long n=1; n<=etot1-1; n++){
//
//
// #   ifdef ON_THE_FLY
//
//     long g_p = LOCAL_GP_NR_OF_EVALP(ar,etot1-1-n); /* grid point nr. of the considered evalpoint */
//     long e_p = g_p;
//
// #   else
//
//     long g_p = GP_NR_OF_EVALP(gridp,ar,etot1-1-n); /* grid point nr. of the considered evalpoint */
//     long e_p = GINDEX(gridp,g_p);
//
// #   endif
//
//
//     if ( evalpoint[e_p].eqp == gridp ){
//
//       u_intensity[RINDEX(g_p,ar)] = (P[n] + P[n-1]) / 2.0;
//     }
//   }


//   u_intensity[RINDEX(gridp,r)]  = (P[etot1] + P[etot1-1]) / 2.0;
//   u_intensity[RINDEX(gridp,ar)] = (P[etot1] + P[etot1-1]) / 2.0;


//   for (long n=etot1+1; n<ndep; n++){
//
//
// #   ifdef ON_THE_FLY
//
//     long g_p = LOCAL_GP_NR_OF_EVALP(r,n-etot1-1);  /* grid point nr. of the considered evalpoint */
//     long e_p = g_p;
//
// #   else
//
//     long g_p = GP_NR_OF_EVALP(gridp,r,n-etot1-1);  /* grid point nr. of the considered evalpoint */
//     long e_p = GINDEX(gridp,g_p);
//
// #   endif
//
//
//     if ( evalpoint[e_p].eqp == gridp ){
//
//       u_intensity[RINDEX(g_p,r)] = (P[n] + P[n-1]) / 2.0;
//     }
//   }
//
//
//   {
//
//
// #   ifdef ON_THE_FLY
//
//     long g_p = LOCAL_GP_NR_OF_EVALP(r,etot2-1);    /* grid point nr. of the considered evalpoint */
//     long e_p = g_p;
//
// #   else
//
//     long g_p = GP_NR_OF_EVALP(gridp,r,etot2-1);    /* grid point nr. of the considered evalpoint */
//     long e_p = GINDEX(gridp,g_p);
//
// #   endif
//
//
//     if ( evalpoint[e_p].eqp == gridp ){
//
//       u_intensity[RINDEX(g_p,r)] = P[ndep-1];
//     }
//   }


  /*_____________________________________________________________________________________________*/


  /* Free the allocated memory for temporary variables */

  free( A );
  free( C );
  free( F );
  free( G );


  return(0);

}

/*-----------------------------------------------------------------------------------------------*/
