/* Frederik De Ceuster - University College London                                               */
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

#include "declarations.hpp"
#include "exact_feautrier.hpp"



/* exact_feautrier: fill Feautrier matrix, solve exactly, return (P[etot1-1]+P[etot1-2])/2       */
/*-----------------------------------------------------------------------------------------------*/

int exact_feautrier( long ndep, double *S, double *dtau, long etot1, long etot2, double ibc,
                     EVALPOINT *evalpoint, double *P_intensity, long gridp, long r, long ar )
{


  double *A;                                /* A coefficient in the Feautrier recursion relation */
  A = (double*) malloc( ndep*sizeof(double) );

  double *B;                                /* B coefficient in the Feautrier recursion relation */
  B = (double*) malloc( ndep*sizeof(double) );

  double *C;                                /* C coefficient in the Feautrier recursion relation */
  C = (double*) malloc( ndep*sizeof(double) );

  double *Fd1;                             /* upper-tridiagonal elements of the Feautrier matrix */
  Fd1 = (double*) malloc( ndep*sizeof(double) );

  double *Fd2;                                      /* diagonal elements of the Feautrier matrix */
  Fd2 = (double*) malloc( ndep*sizeof(double) );

  double *Fd3;                             /* lower-tridiagonal elements of the Feautrier matrix */
  Fd3 = (double*) malloc( ndep*sizeof(double) );

  double *P;                                                         /* Feautrier mean intensity */
  P = (double*) malloc( ndep*sizeof(double) );

  double *D;                                                                  /* helper variable */
  D = (double*) malloc( ndep*sizeof(double) );





  /*   SETUP THE FEAUTRIER RECURSION RELATION                                                    */
  /*_____________________________________________________________________________________________*/


  /* Define the Feautrier A, B and C */

  A[0]      = 1.0 / dtau[1];
  A[ndep-1] = 1.0 / dtau[ndep-1];


  for (long n=1; n<ndep-1; n++) {

    A[n] = 2.0 / (dtau[n] + dtau[n+1]) / dtau[n];
    C[n] = 2.0 / (dtau[n] + dtau[n+1]) / dtau[n+1];

    B[n] = 1.0 + A[n] + C[n];
  }


  /* Define the Feautrier matrix */
  /* and store the source function S initially in P */

  for (long n=1; n<ndep-1; n++){

    Fd1[n] = -A[n];
    Fd2[n] =  B[n] - 1.0;                             /* subtract the 1.0 for numerical purposes */
    Fd3[n] = -C[n];

    P[n]   =  S[n];
  }


  /* Define boundary conditions for the Feautrier matrix */

  Fd1[0] =  0.0;
  Fd2[0] =  2.0*A[0] + 2.0*A[0]*A[0];
  Fd3[0] = -2.0*A[0]*A[0];

  P[0]   = S[0] + 2.0*ibc*exp(-dtau[0]) / dtau[1];


  Fd1[ndep-1] = -2.0*A[ndep-1]*A[ndep-1];
  Fd2[ndep-1] = 2.0*A[ndep-1] + 2.0*A[ndep-1]*A[ndep-1];
  Fd3[ndep-1] = 0.0;

  P[ndep-1]   = S[ndep-1] + 2.0*ibc*exp(-dtau[0]) / dtau[ndep-1];


  /*_____________________________________________________________________________________________*/





  /*   SOLVE THE FEAUTRIER RECURSION RELATION                                                    */
  /*_____________________________________________________________________________________________*/


  /* Elimination step */

  double bet  = 1.0 + Fd2[0];

  P[0] = P[0] / bet;


  for (long n=1; n<ndep; n++){

    D[n] = Fd3[n-1] / bet;
    bet  = 1.0 + (Fd2[n] - (Fd1[n]*D[n]));    /* add 1.0, after the large numbers are subtracted */
    P[n] = (P[n] - Fd1[n]*P[n-1]) / bet;
  }


  /* Back substitution */

  for (long n=ndep-2; n>=0; n--){

    P[n] = P[n] - D[n+1]*P[n+1];
  }


  /*_____________________________________________________________________________________________*/





  /*   RELATE THE RESULTS BACK TO THE EVALUATION POINTS VIA P_intensity                          */
  /*_____________________________________________________________________________________________*/


  // printf("(exact_feautrier): for point %ld and ray %ld \n", gridp, r);

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
  free( B );
  free( C );
  free( Fd1 );
  free( Fd2 );
  free( Fd3 );
  free( P );
  free( D );


  return(0);

}

/*-----------------------------------------------------------------------------------------------*/
