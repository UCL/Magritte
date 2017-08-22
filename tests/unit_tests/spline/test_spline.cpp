/* Frederik De Ceuster - University College London                                               */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/* test_spline: Calculate splines for tabulated functions                                        */
/*                                                                                               */
/* (based on spline in 3D-PDR)                                                                   */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/


#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "catch.hpp"

#include "../../../src/spline.hpp"


#define EPS 1.0E-2                                               /* error allowed in computation */



/* Some example functions to approximate                                                         */
/*-----------------------------------------------------------------------------------------------*/

double f_test(double x)
{

  return 2.0*(x-0.1)*(x-0.2)*(x-0.35);
}

/*-----------------------------------------------------------------------------------------------*/



/* Some example functions to approximate                                                         */
/*-----------------------------------------------------------------------------------------------*/

// double f_test(double x)
// {
//
//   return 2.0/(x+0.1) + 5*exp(-(x-2)*(x-2)/0.003);
// }

/*-----------------------------------------------------------------------------------------------*/





/* Test spline interpolations                                                                    */
/*-----------------------------------------------------------------------------------------------*/

TEST_CASE("Spline interpolations"){


  long i;                                                                               /* index */

  long n = 20;                                                          /* length of the vectors */

  double xa[n];                                            /* x-values of the tabulated function */
  double ya[n];                                            /* y-values of the tabulated function */


  double yp0 = 1.0E30;                                               /* lower boundary condition */
  double ypn = 1.0E30;                                               /* upper boundary condition */

  double d2ya[n];                                     /* second order derivative of the function */


  for (i=0; i<n; i++){

    xa[i] = 0.2*i;
    ya[i] = f_test(xa[i]);
  }



  /* Test spline and splint */

  spline( xa, ya, n, yp0, ypn, d2ya );


  long n_extra = 7*n;

  double x_test[n_extra];
  double y_test[n_extra];


  for (i=0; i<n_extra; i++){

    x_test[i] = (xa[n-1]-xa[0])/n_extra*i + xa[0];

    splint( xa, ya, d2ya, n, x_test[i], &y_test[i]);
  }


  /* Test the accuracy of the interpolation */

  for (i=0; i<n_extra; i++){

    CHECK( Approx(y_test[i]).epsilon(EPS) == f_test(x_test[i]) );
  }



  /* Write the functions to a text file for plotting */

  FILE *table = fopen("output/test_spline_table.txt", "w");

  if (table == NULL){

      printf("Error opening file!\n");
      exit(1);
    }

  for (i=0; i<n; i++){

    fprintf(table, "%lE\t%lE\t%lE\n", xa[i], ya[i], d2ya[i] );
  }

  fclose(table);


  FILE *func = fopen("output/test_spline_func.txt", "w");

  if (func == NULL){

      printf("Error opening file!\n");
      exit(1);
    }

  for (i=0; i<n_extra; i++){

    fprintf(func, "%lE\t%lE\n", x_test[i], y_test[i] );
  }

  fclose(func);

}

/*-----------------------------------------------------------------------------------------------*/
