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



/* feautrier: fill Feautrier matrix, solve exactly, return (P[etot1-1]+P[etot1-2])/2             */
/*-----------------------------------------------------------------------------------------------*/

int feautrier( EVALPOINT *evalpoint, long gridp, long r, long ar, double *S, double *dtau,
               double *P_intensity, double *Lambda_diagonal )
{


  long etot1 = raytot[RINDEX(gridp, ar)];      /* total number of evaluation points along ray ar */

  long etot2 = raytot[RINDEX(gridp, r)];        /* total number of evaluation points along ray r */

  long ndep = etot1 + etot2;               /* nr. of depth points along a pair of antipodal rays */


  double *A;                                    /* A coefficient in Feautrier recursion relation */
  A = (double*) malloc( ndep*sizeof(double) );

  double *C;                                    /* C coefficient in Feautrier recursion relation */
  C = (double*) malloc( ndep*sizeof(double) );

  double *F;                                                                  /* helper variable */
  F = (double*) malloc( ndep*sizeof(double) );

  double *G;                                                                  /* helper variable */
  G = (double*) malloc( ndep*sizeof(double) );

  double *P;                                                         /* Feautrier mean intensity */
  P = (double*) malloc( ndep*sizeof(double) );





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


  /* Store the source function S initially in P */

  for (long n=0; n<ndep; n++){

    P[n] = S[n];
  }


  /*_____________________________________________________________________________________________*/





  /*   SOLVE THE FEAUTRIER RECURSION RELATION                                                    */
  /*_____________________________________________________________________________________________*/


  /* Elimination step */

  P[0] = P[0] / B0;
  F[0] = (B0 - C[0])/ C[0];


  for (long n=1; n<ndep-1; n++){

    F[n] = ( 1.0 + A[n]*F[n-1]/(1.0 + F[n-1]) ) / C[n];

    P[n] = (P[n] + A[n]*P[n-1]) / (1.0 + F[n]) / C[n];
  }


  P[ndep-1] = (P[ndep-1] + A[ndep-1]*P[ndep-2]) / (Bndepm1 - A[ndep-1]/(1.0 + F[ndep-2]) );

  G[ndep-1] = (Bndepm1 - A[ndep-1]) / A[ndep-1];


  /* Back substitution */

  for (long n=ndep-2; n>0; n--){

    P[n] = P[n] + P[n+1]/(1.0+F[n]);

    G[n] = (1.0 + C[n]*G[n+1]/(1.0+G[n+1])) / A[n];
  }


  P[0] = P[0] + P[1]/(1.0+F[0]);


  /*_____________________________________________________________________________________________*/



  // for (int n=0; n<ndep; n++){
  //
  //   printf("%d   %lE   %lE   %lE\n", n, S[n], dtau[n],  P[n] );
  // }



  /*   CALCULATE THE LAMBDA OPERATOR                                                             */
  /*_____________________________________________________________________________________________*/


  Lambda_diagonal[0] = 1.0 / (B0 - C[0]/(1.0+G[1]));


  for (long n=1; n<ndep-1; n++){

    Lambda_diagonal[n] = (1.0 + G[n+1]) / (F[n-1] + G[n+1] + F[n-1]*G[n+1]) / C[n];
  }


  Lambda_diagonal[ndep-1] = 1.0 / (Bndepm1 - A[ndep-1]/(1.0+F[ndep-2]));


  /*_____________________________________________________________________________________________*/





  /*   RELATE THE RESULTS BACK TO THE EVALUATION POINTS VIA P_intensity                          */
  /*_____________________________________________________________________________________________*/


  // printf("(feautrier): for point %ld and ray %ld \n", gridp, r);

  {
    long g_p = GP_NR_OF_EVALP(gridp,ar,etot1-1);   /* grid point nr. of the considered evalpoint */

    if ( evalpoint[GINDEX(gridp,g_p)].eqp == gridp ){

      P_intensity[RINDEX(g_p,ar)] = P[0];

      // printf( "In! ar for %ld is %lE\n", g_p, P_intensity[RINDEX(g_p,ar)] );
    }
  }


  for (long n=1; n<=etot1-1; n++){

    long g_p = GP_NR_OF_EVALP(gridp,ar,etot1-1-n); /* grid point nr. of the considered evalpoint */

    if ( evalpoint[GINDEX(gridp,g_p)].eqp == gridp ){

      P_intensity[RINDEX(g_p,ar)] = (P[n] + P[n-1]) / 2.0;

      // printf( "In! ar for %ld is %lE\n", g_p, P_intensity[RINDEX(g_p,ar)] );
    }
  }


  P_intensity[RINDEX(gridp,ar)] = (P[etot1] + P[etot1-1]) / 2.0;
  P_intensity[RINDEX(gridp,r)]  = (P[etot1] + P[etot1-1]) / 2.0;


  for (long n=etot1+1; n<ndep; n++){

    long g_p = GP_NR_OF_EVALP(gridp,r,n-etot1-1);  /* grid point nr. of the considered evalpoint */

    if ( evalpoint[GINDEX(gridp,g_p)].eqp == gridp ){

      P_intensity[RINDEX(g_p,r)] = (P[n] + P[n-1]) / 2.0;

      // printf( "In! r for %ld is %lE\n", g_p, P_intensity[RINDEX(g_p,r)] );
    }
  }


  {
    long g_p = GP_NR_OF_EVALP(gridp,r,etot2-1);    /* grid point nr. of the considered evalpoint */

    if ( evalpoint[GINDEX(gridp,g_p)].eqp == gridp ){

      P_intensity[RINDEX(g_p,r)] = P[ndep-1];

      // printf( "In! r for %ld is %lE\n", g_p, P_intensity[RINDEX(g_p,r)] );
    }
  }

  /*_____________________________________________________________________________________________*/





  /* Write the source function and opacity at each point to a file (only for testing) */


    // for (long n=0; n<ndep; n++){
    //
    //   printf(" for gridp %ld we have S= %lE , dtau= %lE , P= %lE\n", gridp, S[n], dtau[n], P[n]);
    // }

    // for (long gp=0; gp<NGRID; gp++){
    //   for(long ray=0; ray<NRAYS; ray++){
    //
    //     printf("%.2lE \t", P_intensity[RINDEX(gp,ray)]);
    //   }
    //   printf("\n");
    // }

  // FILE *SandOP = fopen("output/SandOP.txt", "w");

  // if (SandOP == NULL){

  //   printf("Error opening file!\n");
  //   exit(1);
  // }


  // for (long n=0; n<ndep; n++){

  //   fprintf( SandOP, "%lE\t%lE\t%lE\t%lE\t%lE\t%lE\n",
  //            Fd1[n], Fd2[n], Fd3[n], S[n], dtau[n], P[n] );
  // }

  // fclose(SandOP);



  free( A );
  free( C );
  free( F );
  free( G );
  free( P );


  return(0);

}

/*-----------------------------------------------------------------------------------------------*/
