/* Frederik De Ceuster - University College London                                               */
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



/* spline: calculate the cubic spline (cfr. Numerical Recipes, Chapter 3.3: Spline Routine)      */
/*-----------------------------------------------------------------------------------------------*/

void spline( double *x, double *y, long n, double yp0, double ypn, double *d2y )
{

  /* x   = vector for independent variable
     y   = vector for x-dependent variable
     n   = length of the x and y vector
     yp0 = first derivative of the function at point 0
     ypn = first derivative of the function at point n-1
     d2y = vector for the second derivative of the function */


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

    d2y[i] = d2y[i]*d2y[i] + u[i];
  }


  free(u);

}

/*-----------------------------------------------------------------------------------------------*/





/* splint: spline interpolations                                                                 */
/*-----------------------------------------------------------------------------------------------*/

void splint(double *xa, double *ya, double *d2ya, long n, double x, double *y)
{

  /* xa = vector for the independent variable
     ya = vector for the x-dependent variable
     d2ya = vector for the second derivative of the function
     n = lengths of the previous vectors
     x = x-valua at which y is to be interpolated
     y = result of the interpolation */


  long j_lo = 0;
  long j_hi = n-1;
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


  if ( ((x < xa[0])  &&  ascending)  ||  ((x > xa[0])  &&  !ascending) ){

    j_lo = 0;
    j_hi = 1;
  }

  if ( ((x > xa[n-1])  &&  ascending)  ||  ((x < xa[n-1])  &&  !ascending) ){

    j_lo = n-2;
    j_hi = n-1;
  }



  /* Evaluate the cubic spline polynomial */  

  double A = (xa[j_hi]-x) / (xa[j_hi]-xa[j_lo]);

  double B = (x-xa[j_lo]) / (xa[j_hi]-xa[j_lo]);

  *y = A*ya[j_lo] + B*ya[j_hi]
       + ( (pow(A,3)-A)*d2ya[j_lo]
            + (pow(B,3)-B)*d2ya[j_hi] ) * pow(xa[j_hi]-xa[j_lo],2) / 6.0;

}

/*-----------------------------------------------------------------------------------------------*/