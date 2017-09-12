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

#include "../../../src/declarations.hpp"
#include "../../../src/definitions.hpp"
#include "../../../src/initializers.hpp"
#include "../../../src/read_input.hpp"
#include "../../../src/create_healpixvectors.hpp"
#include "../../../src/ray_tracing.hpp"
#include "../../../src/exact_feautrier.hpp"


#define EPS 1.0E-9



TEST_CASE("Feautrier solver"){





  /*   SET UP TEST DATA                                                                          */
  /*_____________________________________________________________________________________________*/


  double unit_healpixvector[3*NRAYS];            /* array of HEALPix vectors for each ipix pixel */

  long   antipod[NRAYS];                                     /* gives antipodal ray for each ray */


  /* Define grid (using types defined in definitions.h) */

  GRIDPOINT gridpoint[NGRID];                                                     /* grid points */

  EVALPOINT evalpoint[NGRID*NGRID];                     /* evaluation points for each grid point */


  /* Since the executables are now in the directory /tests, we have to change the paths */

  string test_grid_inputfile   = "../../../" + grid_inputfile;


  initialize_evalpoint(evalpoint);


  /* Initialize the data structures which will store the evaluation pointa */

  initialize_long_array(key, NGRID*NGRID);

  initialize_long_array(raytot, NGRID*NRAYS);

  initialize_long_array(cum_raytot, NGRID*NRAYS);


  /* Read input file */

  read_input(test_grid_inputfile, gridpoint);


  /* Setup the (unit) HEALPix vectors */

  create_healpixvectors(unit_healpixvector, antipod);


  /* Ray tracing */

  ray_tracing(unit_healpixvector, gridpoint, evalpoint);



  /* Read the test data (source and optical depth increments) */

  char buffer[BUFFER_SIZE];                                         /* buffer for a line of data */

  string testdata0 = "test_data/intens_0.dat";
  string testdata1 = "test_data/intens_1.dat";

  long ndep=NGRID-1;

  double S[ndep];

  double dtau[ndep];

  double P_test[ndep];



  /* Define and initialize the resulting P_intensity array */

  double P_intensity[NGRID*NRAYS];                       /* Feautrier's mean intensity for a ray */

  initialize_double_array(P_intensity, NGRID*NRAYS);

  long r=0;

  long ar=10;


  /*_____________________________________________________________________________________________*/





  SECTION("Compare with fortran results in intens_0"){


    /* Read input file */

    FILE *data0 = fopen(testdata0.c_str(), "r");


    for (int n=0; n<ndep; n++){

      fgets( buffer, BUFFER_SIZE, data0 );

      sscanf( buffer, "%lf\t%lf\t%lf",
              &(S[n]), &(dtau[n]), &(P_test[n]) );
    }


    fclose(data0);


    double ibc = 0.0;


    /* Check the directly returned values */

    for (int n=1; n<ndep; n++){

      long  etot1 = raytot[RINDEX(n, ar)];
      long  etot2 = raytot[RINDEX(n, r)];

      exact_feautrier(ndep,S,dtau,etot1,etot2, ibc, evalpoint,P_intensity,n,r,ar);

      CHECK( (P_test[n]+P_test[n-1])/2.0/P_intensity[RINDEX(n,r)] == Approx( 1.0 ).epsilon(EPS) );

    }


    /* Check values stored in P_intensity */

    // long etot1 = raytot[RINDEX(2, ar)];
    // long etot2 = raytot[RINDEX(2, r)];
    //
    // exact_feautrier(ndep,S,dtau,etot1,etot2,evalpoint,P_intensity,2,r,ar);
    //
    //
    // for (int n=0; n<etot1; n++){
    //
    //   printf("%lE\t%lE\t%lE\n", S[n], dtau[n], P_intensity[RINDEX(n,ar)]);
    // }
    //
    // for (int n=0; n<etot2; n++){
    //
    //   printf("%lE\t%lE\t%lE\n", S[n], dtau[n], P_intensity[RINDEX(n,r)]);
    // }


  }


  SECTION("Compare with fortran results in /test_data"){


    /* Read input file */

    FILE *data1 = fopen(testdata1.c_str(), "r");


    for (int n=0; n<ndep; n++){

      fgets( buffer, BUFFER_SIZE, data1 );

      sscanf( buffer, "%lf\t%lf\t%lf",
              &(S[n]), &(dtau[n]), &(P_test[n]) );
    }


    fclose(data1);


    double ibc = 4.4718814518123E-19;


    /* Check the directly returned values */

    for (int n=1; n<ndep; n++){

      long  etot1 = raytot[RINDEX(n, ar)];
      long  etot2 = raytot[RINDEX(n, r)];

      exact_feautrier(ndep,S,dtau,etot1,etot2, ibc, evalpoint,P_intensity,n,r,ar);

      CHECK( (P_test[n]+P_test[n-1])/2.0/P_intensity[RINDEX(n,r)] == Approx( 1.0 ).epsilon(EPS) );

    }


    /* Check values stored in P_intensity */

    // long etot1 = raytot[RINDEX(2, ar)];
    // long etot2 = raytot[RINDEX(2, r)];
    //
    // exact_feautrier(ndep,S,dtau,etot1,etot2,evalpoint,P_intensity,2,r,ar);
    //
    //
    // for (int n=0; n<etot1; n++){
    //
    //   printf("%lE\t%lE\t%lE\n", S[n], dtau[n], P_intensity[RINDEX(n,ar)]);
    // }
    //
    // for (int n=0; n<etot2; n++){
    //
    //   printf("%lE\t%lE\t%lE\n", S[n], dtau[n], P_intensity[RINDEX(n,r)]);
    // }


  }


}