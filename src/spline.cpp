/* Frederik De Ceuster - University College London & KU Leuven                                   */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/* spline: Calculate splines for tabulated functions                                             */
/*                                                                                               */
/* (based on spline in 3D-PDR)                                                                   */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/



#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "spline.hpp"

#define IND(r,c) ((c)+(r)*n)



/* spline: calculate the cubic spline (cfr. Numerical Recipes, Chapter 3.3: Spline Routine)      */
/*-----------------------------------------------------------------------------------------------*/

void spline( double *x, double *y, long n, double yp0, double ypn, double *d2y )
{

  /*  Given the arrays x and y (size n) containing a tabulated function, i.e., y[i]=f(x[i]),
      with x[0] < x[2] < ... < x[n-1], and given values yp0 and ypn for the first derivative
      of the interpolating function at points 0 and n-1, respectively, this routine returns an
      array d2y of length n, which contains the second derivatives of the interpolating function
      at the tabulated points x[i]. If yp0 and/or ypn are equal to 1.0E+30 or larger, the routine
      is signalled to set the corresponding boundary condition for a natural spline, with zero
      second derivative at that boundary.

      x   = vector for independent variable
      y   = vector for x-dependent variable
      n   = length of the x and y vector
      yp0 = first derivative of the function at point 0
      ypn = first derivative of the function at point n-1
      d2y = vector for the second derivative of the function  */


  long i;                                                                               /* index */

  double sig;
  double p;
  double qn;
  double un;

  double *u;
  u = (double*) malloc( n*sizeof(double) );



  /* Set lower boundary conditions */

  if (yp0 >= 1.0E30){


    /* invoke the "natural" lower boundaty condition */

    d2y[0] = 0.0;
    u[0]   = 0.0;
  }

  else {


    /* give specified first derivative */

    d2y[0] = -0.5;
    u[0]   = (3.0/(x[1]-x[0])) * ((y[1]-y[0])/(x[1]-x[0]) - yp0);
  }


  for (i=1; i<n-1; i++){

    sig = (x[i]-x[i-1]) / (x[i+1]-x[i-1]);

    p = sig*d2y[i-1] + 2.0;

    d2y[i] = (sig-1.0) / p;

    u[i] = ( 6.0* ( (y[i+1]-y[i])/(x[i+1]-x[i]) - (y[i]-y[i-1])/(x[i]-x[i-1]) )
                    / (x[i+1]-x[i-1]) - sig*u[i-1] ) / p;
  }



  /* Set upper boundary conditions */

  if (ypn >= 1.0E30){


    /* invoke the "natural" upper boundaty condition */

    qn = 0.0;
    un = 0.0;
  }

  else {


    /* give specified first derivative */

    qn = 0.5;
    un = (3.0/(x[n-1]-x[n-2])) * (ypn - (y[n-1]-y[n-2])/(x[n-1]-x[n-2]));
  }


  d2y[n-1] = (un-qn*u[n-2]) / (qn*d2y[n-2] + 1.0);



  /* Back-substitution */

  for (i=n-2; i>=0; i--){

    d2y[i] = d2y[i]*d2y[i+1] + u[i];
  }


  free(u);

}

/*-----------------------------------------------------------------------------------------------*/





/* splint: spline interpolations                                                                 */
/*-----------------------------------------------------------------------------------------------*/

void splint(double *xa, double *ya, double *d2ya, long n, double x, double *y)
{

  /*  Given the arrays xa and ya (size n) containing a tabulated function, i.e., ya[i] = f(xa[i]),
      with the xa[i]'s in order, and given the array d2ya produced by the spline() function, this
      function returns a cubic spline interpolated value y.

      xa   = vector for the independent variable
      ya   = vector for the x-dependent variable
      d2ya = vector for the second derivative of the function
      n    = lengths of the previous vectors
      x    = x-valua at which y is to be interpolated
      y    = result of the interpolation  */


  long j_lo = -1;
  long j_hi = n;
  long j_mid;

  bool ascending = (xa[n-1] > xa[0]);


  /* Find the interval x[j_lo] <= x <= x[j_hi] using bisection method */

  while ( (j_hi-j_lo) > 1 ){

    j_mid = (j_hi+j_lo) / 2;

    if ( (x>xa[j_mid]) == ascending ){

      j_lo = j_mid;
    }

    else {

      j_hi = j_mid;
    }
  } /* end of while loop */


  if ( j_lo == -1 ){

    j_lo = 0;
    j_hi = 1;
  }

  if ( j_lo == n-1 ){

    j_lo = n-2;
    j_hi = n-1;
  }

  // if ( ((x <= xa[0])  &&  ascending)  ||  ((x >= xa[0])  &&  !ascending) ){
  //
  //   j_lo = 0;
  //   j_hi = 1;
  // }
  //
  // if ( ((x >= xa[n-1])  &&  ascending)  ||  ((x <= xa[n-1])  &&  !ascending) ){
  //
  //   j_lo = n-2;
  //   j_hi = n-1;
  // }



  /* Evaluate the cubic spline polynomial */

  double A = (xa[j_hi]-x) / (xa[j_hi]-xa[j_lo]);

  double B = (x-xa[j_lo]) / (xa[j_hi]-xa[j_lo]);

  *y = A*ya[j_lo] + B*ya[j_hi]
       + ( (pow(A,3)-A)*d2ya[j_lo]
            + (pow(B,3)-B)*d2ya[j_hi] ) * pow(xa[j_hi]-xa[j_lo],2) / 6.0;

}

/*-----------------------------------------------------------------------------------------------*/





/* splie2: calculate the cubic splines of the rows for a 2-variable function                     */
/*-----------------------------------------------------------------------------------------------*/

void splie2( double *x1a, double *x2a, double *ya, long m, long n, double *d2ya )
{

  /*  Given a tabulated function ya (of size mxn) and tabulated independent variables x1a
      (m values) and x2a (n values), this routine constructs one-dimensional natural cubic
      splines of the rows of ya and returns the second derivatives in the array d2ya.

      x1a  = first vector for independent variable
      x2a  = second vector for independent variable
      ya   = matrix for x1a and x2a-dependent variable
      m    = length of the x1a vector
      n    = length of the x2a vector
      d2ya = matrix for the second derivative of the function  */


  long i, j;                                                                          /* indices */

  double *ya_temp;
  ya_temp = (double*) malloc( n*sizeof(double) );

  double *d2ya_temp;
  d2ya_temp = (double*) malloc( n*sizeof(double) );


  double yp0 = 1.0E30;        /* Values higher than or equal to 1.0D30 indicate a natural spline */
  double ypn = 1.0E30;        /* Values higher than or equal to 1.0D30 indicate a natural spline */


  for (i=0; i<m; i++){

    for (j=0; j<n; j++){

      ya_temp[j] = ya[IND(i,j)];
    }


    spline( x2a, ya_temp, n, yp0, ypn, d2ya_temp );


    for (j=0; j<n; j++){

      d2ya[IND(i,j)] = d2ya_temp[j];
    }

  }


  free(ya_temp);
  free(d2ya_temp);

}

/*-----------------------------------------------------------------------------------------------*/





/* splin2: interpolate function via a bicubic spline                                             */
/*-----------------------------------------------------------------------------------------------*/

void splin2( double *x1a, double *x2a, double *ya, double *d2ya, long m, long n,
             double x1, double x2, double *y )
{

  /*  Given x1a, x2a, ya, m, n (as described in splie2) and d2ya (as produced by that function),
      and given a desired interpolating point (x1,x2), this function returns an interpolated
      function value y by performing a bicubic spline interpolation.  */


  double *ya_temp;
  ya_temp = (double*) malloc( n*sizeof(double) );

  double *d2ya_temp;
  d2ya_temp = (double*) malloc( n*sizeof(double) );

  double *yy_temp;
  yy_temp = (double*) malloc( m*sizeof(double) );

  double *d2yy_temp;
  d2yy_temp = (double*) malloc( m*sizeof(double) );

  double yp0 = 1.0E31;        /* Values higher than or equal to 1.0D30 indicate a natural spline */
  double ypn = 1.0E31;        /* Values higher than or equal to 1.0D30 indicate a natural spline */


  /* Perform m evaluations of the row splines constructed by
!    splie2 using the one-dimensional spline evaluator splint */

  for (long i=0; i<m; i++){

    for (long j=0; j<n; j++){

      ya_temp[j] = ya[IND(i,j)];

      d2ya_temp[j] = d2ya[IND(i,j)];
    }

    splint(x2a, ya_temp, d2ya_temp, n, x2, &yy_temp[i]);
  }



  /* Construct the one-dimensional column spline and evaluate it */

  spline(x1a, yy_temp, m, yp0, ypn, d2yy_temp);

  splint(x1a, yy_temp, d2yy_temp, m, x1, y);


}

/*-----------------------------------------------------------------------------------------------*/
