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

#include <string>
#include <iostream>
using namespace std;

#include <Eigen/Dense>
using namespace Eigen;

#include "catch.hpp"

#include "../../../src/declarations.hpp"
#include "../../../src/definitions.hpp"
#include "../../../src/level_population_solver.hpp"



#define IND(r,c) ((c)+(r)*n)

#define EPS 1.0E-14                                    /* fractional error allowed in computation */



TEST_CASE("Testing the matrix solver"){





  /*   SET UP TEST DATA                                                                          */
  /*_____________________________________________________________________________________________*/


  MatrixXd a(5,5);

  VectorXd b(5);



  /* Give a random matrix to solve */

  a(0,0)=12.3;   a(0,1)=7;     a(0,2)=9.5; a(0,3)=4; a(0,4)=3;
  a(1,0)=2.3E-8; a(1,1)=5.9E1; a(1,2)=6;   a(1,3)=4; a(1,4)=8.0;
  a(2,0)=1;      a(2,1)=2;     a(2,2)=3;   a(2,3)=4, a(2,4)=5;
  a(3,0)=3;      a(3,1)=2;     a(3,2)=5;   a(3,3)=8, a(3,4)=9;
  a(4,0)=1;      a(4,1)=1;     a(4,2)=1;   a(4,3)=1; a(4,4)=1;

  b(0)=0.0; b(1)=0.0; b(2)=0.0; b(3)=0.0; b(4)=1.0;


  /*_____________________________________________________________________________________________*/





  /*   TEST GAUSS JORDAN MATRIX SOLVER                                                           */
  /*_____________________________________________________________________________________________*/


  VectorXd x = a.colPivHouseholderQr().solve(b);

  /*---------------------------------*/


  CHECK( Approx(x(0)).epsilon(EPS) == -15.26408445759533 );
  CHECK( Approx(x(1)).epsilon(EPS) == 0.5985915542404779 );
  CHECK( Approx(x(2)).epsilon(EPS) == 16.0669013491144   );
  CHECK( Approx(x(3)).epsilon(EPS) == 32.126760469431076 );
  CHECK( Approx(x(4)).epsilon(EPS) == -32.52816891519062 );


  /*_____________________________________________________________________________________________*/





}
