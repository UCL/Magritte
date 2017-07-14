/* Frederik De Ceuster - University College London                                               */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/* test_column_density_calculator: tests the column_density_calculator function                  */
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
#include "../src/column_density_calculator.cpp"

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


  /* Trace the rays */

  void ray_tracing( double theta_crit, double ray_separation2, double *unit_healpixvector,
                    GRIDPOINT *gridpoint, EVALPOINT *evalpoint );

  ray_tracing(theta_crit, ray_separation2, unit_healpixvector, gridpoint, evalpoint);



  /*--- TEMPORARY CHEMISTRY ---*/
  /*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/

  int nspec = 10;                                                /* number of (chemical) species */

  double *abundance;                                  /* relative abundances w.r.t. hydrogen (H) */
  abundance = (double*) malloc( nspec*ngrid*sizeof(double) );

  for (int n=0; n<ngrid; n++){

    for (int spec=0; spec<nspec; spec++){

      abundance[SINDEX(n, spec)] = 1.0;
    }
  }

  double *density;
  density = (double*) malloc( ngrid*sizeof(double) );

  for (int n=0; n<ngrid; n++){

    gridpoint[n].density = 10.0;
  }

  /*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/



  /* Test the column_density_calculator */

  double *column_density;               /* column densities for each species, ray and grid point */
  column_density = (double*) malloc( ngrid*nspec*NRAYS*sizeof(double) );


  void column_density_calculator( GRIDPOINT *gridpoint, EVALPOINT *evalpoint, double *abundance,
                                  double *column_density );

  column_density_calculator( gridpoint, evalpoint, abundance, column_density );


  CHECK( 1==1 );




}

