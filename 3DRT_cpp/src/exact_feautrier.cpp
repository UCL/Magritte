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



/* exact_feautrier: fill Feautrier matrix, solve exactly, return (P[etot1-1]+P[etot1-2])/2       */
/*-----------------------------------------------------------------------------------------------*/

double exact_feautrier( long ndep, double *S, double *dtau, long etot1, long etot2,
                        EVALPOINT *evalpoint, double *P_intensity, long gridp, long r1, long ar1 )
{

  long   n;                                                                             /* index */

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

  double IBC = 0.0;                                            /* boundary conditions */

  double bet;

  double J;                                   /* mean intensity at gridpoint under consideration */



  /* Define the Feautrier A, B and C */

  A[0]      = 1.0 / dtau[1];
  A[ndep-1] = 1.0 / dtau[ndep-1];


  for (n=1; n<ndep-1; n++) {

    A[n] = 2.0 / (dtau[n] + dtau[n+1]) / dtau[n];
    C[n] = 2.0 / (dtau[n] + dtau[n+1]) / dtau[n+1];

    B[n] = 1.0 + A[n] + C[n];
  }


  /* Define the Feautrier matrix */
  /* and store the source function S initially in P */

  for (n=1; n<ndep-1; n++){

    Fd1[n] = -A[n];
    Fd2[n] =  B[n];
    Fd3[n] = -C[n];

    P[n]   =  S[n];
  }


  /* Define boundary conditions for the Feautrier matrix */

  Fd1[0] = 0.0;
  Fd2[0] = 1.0 + 2.0*A[0] + 2.0*A[0]*A[0];
  Fd3[0] = -2.0*A[0]*A[0];

  P[0]   = S[0] + 2.0*IBC*exp(-dtau[0]) / dtau[1];


  Fd1[ndep-1] = -2.0*A[ndep-1]*A[ndep-1];
  Fd2[ndep-1] = 1.0 + 2.0*A[ndep-1] + 2.0*A[ndep-1]*A[ndep-1];
  Fd3[ndep-1] = 0.0;

  P[ndep-1]   = S[ndep-1] + 2.0*IBC*exp(-dtau[ndep-1]) / dtau[ndep-1];


  /* Solve the Feautrier recursion relation */
  /* This is done in 2 steps: elimination and back-substitution */


  /* Elimination step */

  bet  = Fd2[0];
  P[0] = P[0] / bet;


  for (n=1; n<ndep; n++){

    D[n] = Fd3[n-1] / bet;
    bet  = Fd2[n] - (Fd1[n]*D[n]);
    P[n] = (P[n] - Fd1[n]*P[n-1]) / bet;
  }


  /* Back substitution */

  for (n=ndep-2; n>=0; n--){

    P[n] = P[n] - D[n+1]*P[n+1];
  }


  /* Relate the results back to the evaluation points */

  // printf("(exact_feautrier): for point %ld and ray %ld \n", gridp, r1);


  if ( evalpoint[GINDEX(gridp,GP_NR_OF_EVALP(gridp,ar1,etot1-1))].eqp == gridp ){
  
    P_intensity[RINDEX(GP_NR_OF_EVALP(gridp,ar1,etot1-1),ar1)] = P[0];
  
   // printf( "In! ar1 for %ld is %lE\n", GP_NR_OF_EVALP(gridp,ar1,etot1-1),
   //         P_intensity[RINDEX(GP_NR_OF_EVALP(gridp,ar1,etot1-1),ar1)] );
  }


  for (n=1; n<etot1-1; n++){

    if ( evalpoint[GINDEX(gridp,GP_NR_OF_EVALP(gridp,ar1,etot1-1-n))].eqp == gridp ){

      P_intensity[RINDEX(GP_NR_OF_EVALP(gridp,ar1,etot1-1-n),ar1)] = (P[n] + P[n-1]) / 2.0;

      // printf( "In! ar1 for %ld is %lE\n", GP_NR_OF_EVALP(gridp,ar1,etot1-1-n),
      //         P_intensity[RINDEX(GP_NR_OF_EVALP(gridp,ar1,etot1-1-n),ar1)] );
    }
  }


  for (n=etot1-1; n<ndep-1; n++){

    if ( evalpoint[GINDEX(gridp,GP_NR_OF_EVALP(gridp,r1,n-etot1+2))].eqp == gridp ){

      P_intensity[RINDEX(GP_NR_OF_EVALP(gridp,r1,n-etot1+2),r1)] = (P[n+1] + P[n]) / 2.0;

      // printf( "In! r1 for %ld is %lE\n", GP_NR_OF_EVALP(gridp,r1,n-etot1+2),
      //         P_intensity[RINDEX(GP_NR_OF_EVALP(gridp,r1,n-etot1+2),r1)]);
    }
  }


  if ( evalpoint[GINDEX(gridp,GP_NR_OF_EVALP(gridp,r1,etot2-1))].eqp == gridp ){
  
    P_intensity[RINDEX(GP_NR_OF_EVALP(gridp,r1,etot2-1),r1)] = P[ndep-1];
  
    // printf( "In! r1 for %ld is %lE\n", GP_NR_OF_EVALP(gridp,r1,etot2-1),
    //         P_intensity[RINDEX(GP_NR_OF_EVALP(gridp,r1,etot2-1),r1)]);
  }


  /* Calculate return value */

  J = (P[etot1-1] + P[etot1-2]) / 2.0;

  // printf("this is P %lE\n",J );



  /* Write the source function and opacity at each point to a file (only for testing) */


    // for (n=0; n<ndep; n++){

    //   printf("%lE\t%lE\n", S[n], P[n]);
    // }


  // FILE *SandOP = fopen("output/SandOP.txt", "w");

  // if (SandOP == NULL){

  //   printf("Error opening file!\n");
  //   exit(1);
  // }


  // for (n=0; n<ndep; n++){
      
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


  return J;

}

/*-----------------------------------------------------------------------------------------------*/
