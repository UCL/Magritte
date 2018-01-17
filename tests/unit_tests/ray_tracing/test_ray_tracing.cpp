// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#include <string>
#include <iostream>

#include "catch.hpp"

#include "../../../parameters.hpp"
#include "../../../src/Magritte_config.hpp"
#include "../../../src/declarations.hpp"
#include "../../../src/definitions.hpp"

#include "../../../setup/setup_healpixvectors.hpp"
#include "../../../src/initializers.hpp"
#include "../../../src/read_input.hpp"
#include "../../../src/ray_tracing.hpp"
#include "../../../src/write_txt_tools.hpp"

#define EPS 1.0E-5



// typedef struct {
//
//   double x, y, z;                                     /* x, y and z coordinate of the grid point */
//   double vx, vy, vz;             /* x, y and z component of the velocity field of the grid point */
//
//   double density;                                                   /* density at the grid point */
//
//   long neighbor[NRAYS];                                          /* cell numbers of the neighors */
//   long n_neighbors;                                                       /* number of neighbors */
//
// } CELL;



// TEST_CASE("1D regular grid"){
//
//
//
//
//
//   /*   SET UP TEST DATA                                                                          */
//   /*_____________________________________________________________________________________________*/
//
//
//   /* Define grid (using types defined in definitions.h) */
//
//   CELL cell[NCELLS];                                                     /* grid points */
//
//   EVALPOINT evalpoint[NCELLS*NCELLS];                     /* evaluation points for each grid point */
//
//   long key[NCELLS*NCELLS];              /* stores the nrs. of the grid points on the rays in order */
//
//   long raytot[NCELLS*NRAYS];                /* cumulative nr. of evaluation points along each ray */
//
//   long cum_raytot[NCELLS*NRAYS];            /* cumulative nr. of evaluation points along each ray */
//
//
//   /* Since the executables are now in the directory /tests, we have to change the paths */
//
//   // inputfile   = "../../../" + inputfile;
//
//
//   /* IMPORTANT NOTE: THIS IS EXECUTED FOR EVERY SECTION. SINCE THE FILE NAMES ARE EXTERNALLY
//                      DEFINED< THE EXTERA PIECE "../" WILL BE APPENDED FOR EVERY SECTION, HENCE
//                      SEGMENTATION FAULTS WHEN INTRODUCING MULTIPLE SECTIONS                      */
//
//
//   // initialize_evalpoint(evalpoint);
//
//   /* Initialize the data structures which will store the evaluation pointa */
//
//
//   /* Read input file */
//
//   read_input(inputfile, cell);
//
//   write_healpixvectors("");
//
//
//   /*_____________________________________________________________________________________________*/
//
//
//
//
//
//   /*   TEST RAY TRACING                                                                          */
//   /*_____________________________________________________________________________________________*/
//
//
//   ray_tracing(cell, evalpoint, key, raytot, cum_raytot);
//
//   // write_eval("", evalpoint);
//   //
//   // write_key("");
//   //
//   // write_cum_raytot("");
//
//
//   CHECK(true);
//
//   // SECTION( "Ordering tests" ){
//   //
//   //
//   //   /* "Check for zero dZ increments" */
//   //
//   //   for (int n=0; n<NCELLS; n++){
//   //
//   //     for (int r=0; r<NRAYS; r++){
//   //
//   //       for (int e=0; e<raytot[RINDEX(n,r)]; e++){
//   //
//   //         CHECK( evalpoint[GINDEX(n,GP_NR_OF_EVALP(n,r,e))].dZ != 0.0 );
//   //       }
//   //     }
//   //   }
//   //
//   //
//   //
//   //   /* "Check whether all grid points are on a ray (only true in 1D)" */
//   //
//   //   for (int n1=0; n1<NCELLS; n1++){
//   //
//   //     for (int n2=0; n2<NCELLS; n2++){
//   //
//   //       if (n1 != n2){
//   //
//   //         CHECK( evalpoint[GINDEX(n1,n2)].onray == true );
//   //       }
//   //     }
//   //   }
//   //
//   //
//   //
//   //   /* "Check the order of the evaluation points" */
//   //
//   //   for (int n=0; n<NCELLS; n++){
//   //
//   //     for (int r=0; r<NRAYS; r++){
//   //
//   //       for (int e=0; e<raytot[RINDEX(n,r)]; e++){
//   //
//   //         CHECK( Approx(evalpoint[GINDEX(n,GP_NR_OF_EVALP(n,r,e))].Z).epsilon(EPS) == (1.0 + e) );
//   //       }
//   //     }
//   //   }
//   // }
//
//
//   /*_____________________________________________________________________________________________*/
//
//
//
//
//
// }





// #define NRAYS 12
//
// double healpixvector[3*NRAYS];
// long   antipod[NRAYS];


TEST_CASE("Cell structure")
{

  long ncells = 9;

  CELL *cell = new CELL[ncells];

  cell[0].x =  1.0;
  cell[0].y =  1.0;
  cell[0].z =  0.0;

  cell[0].boundary = true;

  cell[1].x = -1.0;
  cell[1].y = -1.0;
  cell[1].z =  0.0;

  cell[1].boundary = true;

  cell[2].x =  0.0;
  cell[2].y =  1.0;
  cell[2].z =  0.0;

  cell[2].boundary = true;

  cell[3].x =  0.0;
  cell[3].y = -1.0;
  cell[3].z =  0.0;

  cell[3].boundary = true;

  cell[4].x =  1.0;
  cell[4].y =  0.0;
  cell[4].z =  0.0;

  cell[4].boundary = true;

  cell[5].x = -1.0;
  cell[5].y =  0.0;
  cell[5].z =  0.0;

  cell[5].boundary = true;

  cell[6].x = -1.0;
  cell[6].y =  1.0;
  cell[6].z =  0.0;

  cell[6].boundary = true;

  cell[7].x =  1.0;
  cell[7].y = -1.0;
  cell[7].z =  0.0;

  cell[7].boundary = true;

  cell[8].x =  0.0;
  cell[8].y =  0.0;
  cell[8].z =  0.0;

  cell[8].boundary = false;


  // setup_healpixvectors(NRAYS, healpixvector, antipod);
  // write_healpixvectors("");

  find_neighbors (ncells, cell);
  find_endpoints (ncells, cell);

  // write_neighbors ("", ncells, cell);


  // for (long c = 0; c < ncells; c++)
  // {
  //   printf("%ld\n", cell[c].n_neighbors);
  //
  //   for (long n = 0; n < cell[c].n_neighbors; n++)
  //   {
  //     printf("cell %ld has neighbors %ld\n", c, cell[c].neighbor[n]);
  //   }
  // }
  //
  // write_healpixvectors("");
  //
  // double dZ = 0.0;
  //
  // long origin = 1;
  // long ray    = 1;
  //
  // double Z = 0.0;
  //
  // long current = origin;
  // long next    = next_cell (ncells, cell, origin, ray, &Z, current, &dZ);
  //
  //
  //
  // while (next != ncells)
  // {
  //   printf("current %ld, next %ld, Z %lE\n", current, next, Z);
  //
  //   current = next;
  //   next    = next_cell (ncells, cell, origin, ray, &Z, current, &dZ);
  // }

  //
  // long origin = 0;
  // long ray    = 5;
  //
  // for (long o = 0; o < ncells; o++){
  //   std::cout << cell[o].Z[ray] << "\n";
  //   std::cout << cell[o].endpoint[ray] << "\n";
  // }

  long o   = 8;
  long ray = 1;

    std::cout << cell[o].Z[ray] << "\n";
    std::cout << cell[o].endpoint[ray] << "\n";

  double dZ = 0.0;
  double Z  = 0.0;

    std::cout << previous_cell (ncells, cell, o, ray, &Z, o, &dZ) << "\n";



  // printf("next %ld,  dZ %lE\n", next, dZ);

  CHECK (true);

}
