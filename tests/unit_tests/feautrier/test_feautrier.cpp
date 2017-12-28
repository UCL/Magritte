/* Frederik De Ceuster - University College London & KU Leuven                                   */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/* test_feautrier: tests the Feautrier solver                                                    */
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

#include "catch.hpp"

#include "parameters.hpp"
#include "../../../src/Magritte_config.hpp"
#include "../../../src/declarations.hpp"

#include "../../../src/definitions.hpp"

#include "../../../src/initializers.hpp"
#include "../../../src/read_input.hpp"
#include "../../../src/create_healpixvectors.hpp"
#include "../../../src/ray_tracing.hpp"
#include "../../../src/feautrier.hpp"
#include "../../../src/write_output.hpp"


#define EPS 1.0E-9



TEST_CASE("Feautrier solver"){





  /*   SET UP TEST DATA                                                                          */
  /*_____________________________________________________________________________________________*/


  double healpixvector[3*NRAYS];            /* array of HEALPix vectors for each ipix pixel */

  long   antipod[NRAYS];                                     /* gives antipodal ray for each ray */


  /* Define grid (using types defined in definitions.h) */

  CELL cell[NCELLS];                                                     /* grid points */

  EVALPOINT evalpoint[NCELLS*NCELLS];                     /* evaluation points for each grid point */

  initialize_evalpoint(evalpoint);


  /* Initialize the data structures which will store the evaluation pointa */

  initialize_long_array(key, NCELLS*NCELLS);

  initialize_long_array(raytot, NCELLS*NRAYS);

  initialize_long_array(cum_raytot, NCELLS*NRAYS);


  /* Since the executables are now in the directory /tests, we have to change the paths */

  std::string test_inputfile   = "../../../" + inputfile;


  /* Read input file */

  read_input(test_inputfile, cell);


  /* Setup the (unit) HEALPix vectors */

  create_healpixvectors(healpixvector, antipod);


  /* Ray tracing */

  ray_tracing(healpixvector, cell, evalpoint);



  /* Read the test data (source and optical depth increments) */

  char buffer[BUFFER_SIZE];                                         /* buffer for a line of data */

  std::string testdata0 = "test_data/intens_0.dat";
  std::string testdata1 = "test_data/intens_1.dat";

  long ndep=NCELLS-1;

  double S[ndep];

  double dtau[ndep];

  double P_test[ndep];



  /* Define and initialize the resulting P_intensity array */

  double P_intensity[NCELLS*NRAYS];                       /* Feautrier's mean intensity for a ray */

  initialize_double_array(P_intensity, NCELLS*NRAYS);

  long r=0;

  long ar=10;


  double Lambda_diagonal[ndep];


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


    feautrier(evalpoint, 0, r, ar, S, dtau, P_intensity, Lambda_diagonal );


    write_double_2("P_intensities", "", ndep, NRAYS, P_intensity);

    for (int n=0; n<ndep; n++){

      feautrier(evalpoint, n, r, ar, S, dtau, P_intensity, Lambda_diagonal );

      // printf("%d   %lE   %lE   %lE\n", n, S[n], dtau[n],  P_intensity[RINDEX(n,r)] );
    }


    /* Check the directly returned values */

    for (int n=1; n<ndep; n++){


      feautrier(evalpoint, n, r, ar, S, dtau, P_intensity, Lambda_diagonal );

      // printf("%lE\n", P_intensity[RINDEX(n,r)] );

      CHECK( (P_test[n]+P_test[n-1])/2.0/P_intensity[RINDEX(n,r)] == Approx( 1.0 ).epsilon(EPS) );

    }


    /* Check values stored in P_intensity */

    long etot1 = raytot[RINDEX(2, ar)];
    long etot2 = raytot[RINDEX(2, r)];

      feautrier(evalpoint, 2, r, ar, S, dtau, P_intensity, Lambda_diagonal );


    for (int n=0; n<etot1; n++){

      printf("%lE\t%lE\t%lE\n", S[n], dtau[n], P_intensity[RINDEX(n,ar)]);
    }

    for (int n=0; n<etot2; n++){

      printf("%lE\t%lE\t%lE\n", S[n], dtau[n], P_intensity[RINDEX(n,r)]);
    }


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



    /* Check the directly returned values */

    for (int n=1; n<ndep; n++){

      long  etot1 = raytot[RINDEX(n, ar)];
      long  etot2 = raytot[RINDEX(n, r)];

      feautrier( evalpoint, n, r, ar, S, dtau, P_intensity, Lambda_diagonal );

      CHECK( (P_test[n]+P_test[n-1])/2.0/P_intensity[RINDEX(n,r)] == Approx( 1.0 ).epsilon(EPS) );

    }


    /* Check values stored in P_intensity */

    // long etot1 = raytot[RINDEX(2, ar)];
    // long etot2 = raytot[RINDEX(2, r)];
    //
    // feautrier(ndep,S,dtau,etot1,etot2,evalpoint,P_intensity,2,r,ar);
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
