/* Frederik De Ceuster - University College London                                               */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/* Header for spline.cpp                                                                         */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/



#ifndef __SPLINE_HPP_INCLUDED__
#define __SPLINE_HPP_INCLUDED__



/* spline: calculate the cubic spline (cfr. Numerical Recipes, Chapter 3.3: Spline Routine)      */
/*-----------------------------------------------------------------------------------------------*/

void spline( double *x, double *y, long n, double yp0, double ypn, double *d2y );

/*-----------------------------------------------------------------------------------------------*/



/* splint: spline interpolations                                                                 */
/*-----------------------------------------------------------------------------------------------*/

void splint(double *xa, double *ya, double *d2ya, long n, double x, double *y);

/*-----------------------------------------------------------------------------------------------*/



/* splie2: calculate the cubic splines of the rows for a 2-variable function                     */
/*-----------------------------------------------------------------------------------------------*/

void splie2( double *x1a, double *x2a, double *ya, long m, long n, double *d2ya );

/*-----------------------------------------------------------------------------------------------*/



/* splin2: interpolate function via a bicubic spline                                             */
/*-----------------------------------------------------------------------------------------------*/

void splin2( double *x1a, double *x2a, double *ya, double *d2ya, long m, long n,
             double x1, double x2, double *y );

/*-----------------------------------------------------------------------------------------------*/



#endif /* __SPLINE_HPP_INCLUDED__ */

/*-----------------------------------------------------------------------------------------------*/