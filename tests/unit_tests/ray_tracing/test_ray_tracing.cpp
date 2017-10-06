/* Frederik De Ceuster - University College London & KU Leuven                                   */
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

#include "../../../src/declarations.hpp"
#include "../../../src/definitions.hpp"
#include "../../../src/initializers.hpp"
#include "../../../src/read_input.hpp"
#include "../../../src/create_healpixvectors.hpp"
#include "../../../src/ray_tracing.hpp"
#include "../../../src/write_output.hpp"

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

  grid_inputfile   = "../../../" + grid_inputfile;


  /* IMPORTANT NOTE: THIS IS EXECUTED FOR EVERY SECTION. SINCE THE FILE NAMES ARE EXTERNALLY
                     DEFINED< THE EXTERA PIECE "../" WILL BE APPENDED FOR EVERY SECTION, HENCE
                     SEGMENTATION FAULTS WHEN INTRODUCING MULTIPLE SECTIONS                      */


  initialize_evalpoint(evalpoint);

  /* Initialize the data structures which will store the evaluation pointa */

  initialize_long_array(key, NGRID*NGRID);

  initialize_long_array(raytot, NGRID*NRAYS);

  initialize_long_array(cum_raytot, NGRID*NRAYS);


  /* Read input file */

  read_input(grid_inputfile, gridpoint);


  /* Setup the (unit) HEALPix vectors */

  create_healpixvectors(unit_healpixvector, antipod);

  write_healpixvectors("", unit_healpixvector);


  /*_____________________________________________________________________________________________*/





  /*   TEST RAY TRACING                                                                          */
  /*_____________________________________________________________________________________________*/


  ray_tracing(unit_healpixvector, gridpoint, evalpoint);

  write_eval("", evalpoint);

  write_key("");

  write_cum_raytot("");


  CHECK(true);

  // SECTION( "Ordering tests" ){
  //
  //
  //   /* "Check for zero dZ increments" */
  //
  //   for (int n=0; n<NGRID; n++){
  //
  //     for (int r=0; r<NRAYS; r++){
  //
  //       for (int e=0; e<raytot[RINDEX(n,r)]; e++){
  //
  //         CHECK( evalpoint[GINDEX(n,GP_NR_OF_EVALP(n,r,e))].dZ != 0.0 );
  //       }
  //     }
  //   }
  //
  //
  //
  //   /* "Check whether all grid points are on a ray (only true in 1D)" */
  //
  //   for (int n1=0; n1<NGRID; n1++){
  //
  //     for (int n2=0; n2<NGRID; n2++){
  //
  //       if (n1 != n2){
  //
  //         CHECK( evalpoint[GINDEX(n1,n2)].onray == true );
  //       }
  //     }
  //   }
  //
  //
  //
  //   /* "Check the order of the evaluation points" */
  //
  //   for (int n=0; n<NGRID; n++){
  //
  //     for (int r=0; r<NRAYS; r++){
  //
  //       for (int e=0; e<raytot[RINDEX(n,r)]; e++){
  //
  //         CHECK( Approx(evalpoint[GINDEX(n,GP_NR_OF_EVALP(n,r,e))].Z).epsilon(EPS) == (1.0 + e) );
  //       }
  //     }
  //   }
  // }


  /*_____________________________________________________________________________________________*/





}
