/* Frederik De Ceuster - University College London                                               */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/* test_ray_tracing: tests the ray_tracing function                                              */
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

#include "../../src/declarations.hpp"
#include "../../src/definitions.hpp"
#include "../../src/read_input.hpp"
#include "../../src/create_healpixvectors.hpp"
#include "../../src/ray_tracing.hpp"

#define EPS 1.0E-5


TEST_CASE("1D regular grid"){





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

  read_input(grid_inputfile, gridpoint);



  /* Setup the (unit) HEALPix vectors */

  create_healpixvectors(unit_healpixvector, antipod);


  /*_____________________________________________________________________________________________*/





  /*   TEST RAY TRACING                                                                          */
  /*_____________________________________________________________________________________________*/


  ray_tracing(unit_healpixvector, gridpoint, evalpoint);



  SECTION( "Ordering tests" ){

    /* "Check for zero dZ increments" */

    for (int n=0; n<NGRID; n++){

      for (int r=0; r<NRAYS; r++){

        for (int e=0; e<raytot[RINDEX(n,r)]; e++){

          CHECK( evalpoint[GINDEX(n,GP_NR_OF_EVALP(n,r,e))].dZ != 0.0 );
        }
      }
    }



    /* "Check whether all grid points are on a ray (only true in 1D)" */

    for (int n1=0; n1<NGRID; n1++){

      for (int n2=0; n2<NGRID; n2++){

        if (n1 != n2){
          CHECK( evalpoint[GINDEX(n1,n2)].onray == true );
        }
      }
    }



    /* "Check the order of the evaluation points" */

    for (int n=0; n<NGRID; n++){

      for (int r=0; r<NRAYS; r++){

        for (int e=0; e<raytot[RINDEX(n,r)]; e++){

          CHECK( Approx(evalpoint[GINDEX(n,GP_NR_OF_EVALP(n,r,e))].Z).epsilon(EPS) == (1.0 + e) );
        }
      }
    }
  }


  /*_____________________________________________________________________________________________*/





}
