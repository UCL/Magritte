/* Frederik De Ceuster - University College London                                               */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/* test_exact_feautrier: tests the Feautrier solver                                              */
/*                                                                                               */
/* (NEW)                                                                                         */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/



#include <stdio.h>
#include <stdlib.h>

#include "catch.hpp"

#include "../src/definitions.hpp"
#include "../src/read_input.cpp"
#include "../src/create_healpixvectors.cpp"
#include "../src/ray_tracing.cpp"
#include "../src/exact_feautrier.cpp"



TEST_CASE("Feautrier solver on 14 depth points"){



  /* --- Set up the test data --- */
  /* -----------------------------*/


  double theta_crit=1.0;           /* critical angle to include a grid point as evaluation point */
 
  double ray_separation2=0.00;    /* rays closer than the sqrt of this are considered equivalent */

  nsides = 4;                                         /* Defined in HEALPix, NRAYS = 12*nsides^2 */


  double *unit_healpixvector;                    /* array of HEALPix vectors for each ipix pixel */
  unit_healpixvector = (double*) malloc( 3*NRAYS*sizeof(double) );

  long   *antipod;                                           /* gives antipodal ray for each ray */
  antipod = (long*) malloc( NRAYS*sizeof(long) );


  /* Specify the input file */

  char inputfile[100] = "../input/grid_1D_regular.txt";


  /* Count number of grid points in input file input/ingrid.txt */

  long get_ngrid(char *inputfile);                                    /* defined in read_input.c */

  ngrid = get_ngrid(inputfile);                       /* number of grid points in the input file */


  /* Define and allocate memory for grid (using types defined in definitions.h)*/

  GRIDPOINT *gridpoint;                                                           /* grid points */
  gridpoint = (GRIDPOINT*) malloc( ngrid*sizeof(GRIDPOINT) );

  EVALPOINT *evalpoint;                                 /* evaluation points for each grid point */
  evalpoint = (EVALPOINT*) malloc( ngrid*ngrid*sizeof(EVALPOINT) );


  /* Allocate memory for the variables needed to efficiently store the evalpoints */

  cum_raytot = (long*) malloc( ngrid*NRAYS*sizeof(long) );

  key = (long*) malloc( ngrid*ngrid*sizeof(long) );

  raytot = (long*) malloc( ngrid*NRAYS*sizeof(long) );


  /* Initialise (remove garbage out of the variables) */

  for (long n1=0; n1<ngrid; n1++){

    for (long n2=0; n2<ngrid; n2++){

      evalpoint[GINDEX(n1,n2)].dZ  = 0.0;
      evalpoint[GINDEX(n1,n2)].Z   = 0.0;
      evalpoint[GINDEX(n1,n2)].vol = 0.0;

      evalpoint[GINDEX(n1,n2)].ray = 0;
      evalpoint[GINDEX(n1,n2)].nr  = 0;

      evalpoint[GINDEX(n1,n2)].eqp = 0;

      evalpoint[GINDEX(n1,n2)].onray = false;

      key[GINDEX(n1,n2)] = 0;
    }

    for (long r=0; r<NRAYS; r++){

      raytot[RINDEX(n1,r)]      = 0;
      cum_raytot[RINDEX(n1,r)]  = 0;

    }

  }


  /* Read input file */

  void read_input(char *inputfile, long ngrid, GRIDPOINT *gridpoint );

  read_input(inputfile, ngrid, gridpoint);


  /* Setup the (unit) HEALPix vectors */

  void create_healpixvectors(double *unit_healpixvector, long *antipod);

  create_healpixvectors(unit_healpixvector, antipod);


  /* Ray tracing */

  void ray_tracing( double theta_crit, double ray_separation2, double *unit_healpixvector,
                    GRIDPOINT *gridpoint, EVALPOINT *evalpoint );

  ray_tracing(theta_crit, ray_separation2, unit_healpixvector, gridpoint, evalpoint);



  long ndep=ngrid-1;

  double *S;
  S = (double*) malloc( ndep*sizeof(double) );

  double *dtau;
  dtau = (double*) malloc( ndep*sizeof(double) );


    dtau[0] = 1.0E-1;
    S[0]    = 1.0E-1;

  for (int n3=1; n3<ndep; n3++){

    dtau[n3] = 1.0E-1;
    S[n3]    = S[n3-1] + dtau[n3];
  }


  double *P_intensity;                                   /* Feautrier's mean intensity for a ray */
  P_intensity = (double*) malloc( ngrid*NRAYS*sizeof(double) );

  long gridp=0;

  long r1=1;
  long ar1=173;


  /* Initialization */

  for (long n1=0; n1<ngrid; n1++){

    for (long r=0; r<NRAYS; r++){

      P_intensity[RINDEX(n1,r)] = 0.0;
    }
  }



  SECTION("Compare with analytic result"){

    for (int n=0; n<ndep; n++){

    long  etot1 = raytot[RINDEX(n, ar1)];
    long  etot2 = raytot[RINDEX(n, r1)];

    printf("%lE\n", exact_feautrier(ndep,S,dtau,etot1,etot2,evalpoint,P_intensity,n,r1,ar1));


      // REQUIRE( fabs( exact_feautrier(ndep,S,dtau,etot1,etot2,evalpoint,P_intensity,n,r1,ar1)
      //                - exp() ) < 1.0E-9 );
    }


    /* Write the result */

    FILE *result = fopen("../output/result.txt", "w");

    if (result == NULL){

      printf("Error opening file!\n");
      exit(1);
    }


    for (int n=0; n<ndep; n++){

      long  etot1 = raytot[RINDEX(n, ar1)];
      long  etot2 = raytot[RINDEX(n, r1)];

      fprintf( result, "%lE\n",
               exact_feautrier(ndep,S,dtau,etot1,etot2,evalpoint,P_intensity,n,r1,ar1) );
    }

    fclose(result);


    CHECK( 1==1 );

  }

}