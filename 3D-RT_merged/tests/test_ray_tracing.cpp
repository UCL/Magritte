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

#include "catch.hpp"

#include "../src/definitions.hpp"
#include "../src/read_input.cpp"
#include "../src/create_healpixvectors.cpp"
#include "../src/ray_tracing.cpp"

#define EPS 1.0E-7


TEST_CASE("1D regular grid"){



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



  /* -----------------------------*/



  /* --- Test ray_tracing --- */
  /*--------------------------*/

  void ray_tracing( double theta_crit, double ray_separation2, double *unit_healpixvector,
                    GRIDPOINT *gridpoint, EVALPOINT *evalpoint );

  ray_tracing(theta_crit, ray_separation2, unit_healpixvector, gridpoint, evalpoint);

  /*--------------------------*/




  SECTION( "Check for zero dZ increments" ){

    for (int n=0; n<ngrid; n++){

      for (int r=0; r<NRAYS; r++){

        for (int e=0; e<raytot[RINDEX(n,r)]; e++){

          CHECK( evalpoint[GINDEX(n,GP_NR_OF_EVALP(n,r,e))].dZ != 0.0 );
        }
      }
    }
  }


  SECTION( "Check whether all grid points are on a ray (only true in 1D)" ){

    for (int n1=0; n1<ngrid; n1++){

      for (int n2=0; n2<ngrid; n2++){

        if (n1 != n2){
          CHECK( evalpoint[GINDEX(n1,n2)].onray == true );
        }
      }
    }
  }



  SECTION( "Check the order of the evaluation points" ){
   
    for (int n=0; n<ngrid; n++){

      for (int r=0; r<NRAYS; r++){

        for (int e=0; e<raytot[RINDEX(n,r)]; e++){

          CHECK( Approx(evalpoint[GINDEX(n,GP_NR_OF_EVALP(n,r,e))].Z).epsilon(EPS) == (1.0 + e) );
        }
      }
    }
  }


}

