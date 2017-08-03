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
#include <omp.h>

#include <string>
#include <iostream>
using namespace std;

#include "catch.hpp"

#include "../src/definitions.hpp"
#include "../src/read_input.cpp"
#include "../src/create_healpixvectors.cpp"
#include "../src/ray_tracing.cpp"
#include "../src/exact_feautrier.cpp"


#define EPS 1.0E-21



TEST_CASE("Feautrier solver on 14 depth points"){





  /*   SET UP TEST DATA                                                                          */
  /*_____________________________________________________________________________________________*/


  double unit_healpixvector[3*NRAYS];            /* array of HEALPix vectors for each ipix pixel */

  long   antipod[NRAYS];                                     /* gives antipodal ray for each ray */



  /* Define grid (using types defined in definitions.h) */

  GRIDPOINT gridpoint[NGRID];                                                     /* grid points */

  EVALPOINT evalpoint[NGRID*NGRID];                     /* evaluation points for each grid point */



  /* Since the executables are now in the directory /tests, we have to change the paths */

  grid_inputfile   = "../" + grid_inputfile;
  spec_datafile    = "../" + spec_datafile;
  line_datafile[0] = "../" + line_datafile[0];



  /* Initialize */

  for (long n1=0; n1<NGRID; n1++){

    for (long n2=0; n2<NGRID; n2++){

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

  void read_input(string grid_inputfile, GRIDPOINT *gridpoint );

  read_input(grid_inputfile, gridpoint);



  /* Setup the (unit) HEALPix vectors */

  void create_healpixvectors(double *unit_healpixvector, long *antipod);

  create_healpixvectors(unit_healpixvector, antipod);



  /* Ray tracing */

  void ray_tracing( double *unit_healpixvector, GRIDPOINT *gridpoint, EVALPOINT *evalpoint );

  ray_tracing(unit_healpixvector, gridpoint, evalpoint);



  /* Read the test data (source and optical depth increments) */

  char buffer[BUFFER_SIZE];                                         /* buffer for a line of data */

  string testdata = "test_data/intens_1.dat";

  long ndep=NGRID-1;

  double S[ndep];

  double dtau[ndep];

  double P_test[ndep];

  int ind;


  /* Read input file */

  FILE *data = fopen(testdata.c_str(), "r");


  for (int n=0; n<ndep; n++){

    fgets( buffer, BUFFER_SIZE, data );

    sscanf( buffer, "%lf\t%lf\t%lf",
            &(S[n]), &(dtau[n]), &(P_test[n]) );
  }


  fclose(data);


  /* Define and initialize the resulting P_intensity array */

  double P_intensity[NGRID*NRAYS];                       /* Feautrier's mean intensity for a ray */


  for (long n1=0; n1<NGRID; n1++){

    for (long r=0; r<NRAYS; r++){

      P_intensity[RINDEX(n1,r)] = 0.0;
    }
  }


  long gridp=0;

  long r1=1;
  long ar1=173;


  /*_____________________________________________________________________________________________*/





  SECTION("Compare with fortran results in /test_data"){


    /* Check the directly returned values */

    for (int n=1; n<ndep; n++){

      long  etot1 = raytot[RINDEX(n, ar1)];
      long  etot2 = raytot[RINDEX(n, r1)];

      // printf( "etot1 and etot2 are %d and %d with %lE\n", etot1, etot2,
      //         exact_feautrier(ndep,S,dtau,etot1,etot2,evalpoint,P_intensity,n,r1,ar1)/P_test[n] );

      CHECK( exact_feautrier(ndep,S,dtau,etot1,etot2,evalpoint,P_intensity,n,r1,ar1)
             == Approx( (P_test[n]+P_test[n-1])/2.0 ).epsilon(EPS) );

    }



    /* Check values stored in P_intensity */

    long etot1 = raytot[RINDEX(2, ar1)];
    long etot2 = raytot[RINDEX(2, r1)];

    exact_feautrier(ndep,S,dtau,etot1,etot2,evalpoint,P_intensity,2,r1,ar1);


    for (int n=0; n<etot1; n++){

      // printf("%lE\t%lE\t%lE\n", S[n], dtau[n], P_intensity[RINDEX(n,ar1)]);
    }

    for (int n=0; n<etot2; n++){

      // printf("%lE\t%lE\t%lE\n", S[n], dtau[n], P_intensity[RINDEX(n,r1)]);
    }


  }

}
