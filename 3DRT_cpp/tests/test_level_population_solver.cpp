/* Frederik De Ceuster - University College London                                               */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/* test_level_population_solver: tests the level population solver                               */
/*                                                                                               */
/* (NEW)                                                                                         */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "catch.hpp"

#include "../src/definitions.hpp"
#include "../src/level_population_solver.cpp"


#define IND(r,c) ((c)+(r)*n)

#define EPS 1.0E-14                                    /* fractional error allowed in computation */



TEST_CASE("Testing the matrix solver"){


  /* --- Set up the test data --- */
  /* -----------------------------*/


  long n = 5;                                     /* number of the grid point we are considering */

  int m = 1;                                         /* number of the species we are considering */



  /* Give a random matrix to solve */

  double *a;
  a = (double*) malloc( n*n*sizeof(double) );

  a[IND(0,0)]=12.3;   a[IND(0,1)]=7;     a[IND(0,2)]=9.5; a[IND(0,3)]=4; a[IND(0,4)]=3;
  a[IND(1,0)]=2.3E-8; a[IND(1,1)]=5.9E1; a[IND(1,2)]=6;   a[IND(1,3)]=4; a[IND(1,4)]=8.0;
  a[IND(2,0)]=1;      a[IND(2,1)]=2;     a[IND(2,2)]=3;   a[IND(2,3)]=4, a[IND(2,4)]=5;
  a[IND(3,0)]=3;      a[IND(3,1)]=2;     a[IND(3,2)]=5;   a[IND(3,3)]=8, a[IND(3,4)]=9;
  a[IND(4,0)]=1;      a[IND(4,1)]=1;     a[IND(4,2)]=1;   a[IND(4,3)]=1; a[IND(4,4)]=1;


  double *b;
  b = (double*) malloc( n*m*sizeof(double) );

  b[0]=0.0; b[1]=0.0; b[2]=0.0; b[3]=0.0; b[4]=1.0;


  /* Test Gauss Jordan matrix solver */
  /*---------------------------------*/

  void GaussJordan(int n, int m, double *a, double *b);

  GaussJordan(n, m, a, b);

  /*---------------------------------*/


  CHECK( Approx(b[0]).epsilon(EPS) == -15.26408445759533 );
  CHECK( Approx(b[1]).epsilon(EPS) == 0.5985915542404779 );
  CHECK( Approx(b[2]).epsilon(EPS) == 16.0669013491144   );
  CHECK( Approx(b[3]).epsilon(EPS) == 32.126760469431076 );
  CHECK( Approx(b[4]).epsilon(EPS) == -32.52816891519062 );

}