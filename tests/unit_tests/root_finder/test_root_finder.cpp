
#include <math.h>
#include <stdio.h>



#define EPSILON 3.0E-8
#define PREC    5.0E-3
#define tol1    PREC
#define NGRID   1


double f(double T)
{

  return (T - 2.0)*(2.0+sin(T))*(T+3)*(T+1);
}



double find_root()
{


  double temperature_a = 5.0;
  double temperature_b = 0.0;
  double temperature_c = 0.0;

  double thermal_ratio_a = f(temperature_a);
  double thermal_ratio_b = f(temperature_b);
  double thermal_ratio_c = f(temperature_c);

  double temperature_d;
  double temperature_e;


  for(int i=0; i<100; i++){

    /* Update method based on the Van Wijngaarden-Dekker-Brent rootfinding algorithm */
    /* see Numerical Recipes 9.4 */

    if ( (thermal_ratio_b > 0.0 && thermal_ratio_c > 0.0)
         || (thermal_ratio_b < 0.0 && thermal_ratio_c < 0.0) ){

      temperature_c   = temperature_a;

      thermal_ratio_c = thermal_ratio_a;

      temperature_e   = temperature_d   = temperature_b - temperature_a;

    }


    if ( fabs(thermal_ratio_c) < fabs(thermal_ratio_b) ){

      temperature_a   = temperature_b;
      temperature_b   = temperature_c;
      temperature_c   = temperature_a;

      thermal_ratio_a = thermal_ratio_b;
      thermal_ratio_b = thermal_ratio_c;
      thermal_ratio_c = thermal_ratio_a;

    }


    double tolerance = 2.0*EPSILON*fabs(temperature_b)+0.5*tol1;

    double xm = (temperature_c - temperature_b) / 2.0;

    if ( (fabs(xm) <= tolerance) || (thermal_ratio_b == 0.0) ){


      /* Converged !!! */

      return temperature_b;

    }




  }

  return 0.0; // ERROR

}



int update_x( long gridp, double *temperature_a, double *temperature_b, double *temperature_c,
              double *temperature_d, double *temperature_e,
              double *thermal_ratio_a, double *thermal_ratio_b, double *thermal_ratio_c )
{


  double tolerance = 2.0*EPSILON*fabs(temperature_b[gridp])+0.5*tol1;

  double xm = (temperature_c[gridp] - temperature_b[gridp]) / 2.0;


  if ( (fabs(temperature_e[gridp]) >= tolerance)
       && (fabs(thermal_ratio_a[gridp]) > fabs(thermal_ratio_b[gridp])) ){


    /* Attempt inverse quadratic interpolation */

    double s = thermal_ratio_b[gridp] / thermal_ratio_a[gridp];

    double p, q, r;

    if (temperature_a[gridp] == temperature_c[gridp]){

      p = 2.0 * xm * s;
      q = 1.0 - s;
    }

    else {

      q = thermal_ratio_a[gridp] / thermal_ratio_c[gridp];
      r = thermal_ratio_b[gridp] / thermal_ratio_c[gridp];
      p = s*( 2.0*xm*q*(q-r) - (temperature_b[gridp]-temperature_a[gridp])*(r-1.0) );
      q = (q-1.0)*(r-1.0)*(s-1.0);
    }

    if (p > 0.0){ q = -q; }

    p = fabs(p);


    double min1 = 3.0*xm*q - fabs(tolerance*q);
    double min2 = fabs(temperature_e[gridp]*q);


    if (2.0*p < (min1 < min2 ? min1 :  min2)){  /* Accept interpolation */

      temperature_e[gridp] = temperature_d[gridp];
      temperature_d[gridp] = p / q;
    }

    else {  /* Interpolation failed, use bisection */

      temperature_d[gridp] = xm;
      temperature_e[gridp] = temperature_d[gridp];
    }

  }

  else {  /* Bounds decreasing too slowly, use bisection */

    temperature_d[gridp] = xm;
    temperature_e[gridp] = temperature_d[gridp];
  }


  /* Move last best guess to temperature_a */

  temperature_a[gridp]   = temperature_b[gridp];

  thermal_ratio_a[gridp] = thermal_ratio_b[gridp];


  /* Evaluate new trial root */

  if ( fabs(temperature_d[gridp]) > tolerance ){

    temperature_b[gridp] = temperature_b[gridp] + temperature_d[gridp];
  }

  else {

    if (xm > 0.0){ temperature_b[gridp] = temperature_b[gridp] + fabs(tolerance); }

    else { temperature_b[gridp] = temperature_b[gridp] - fabs(tolerance); }
  }


  return(0);

}

int shuffle_x( long gridp, double *temperature_a, double *temperature_b, double *temperature_c,
               double *temperature_d, double *temperature_e,
               double *thermal_ratio_a, double *thermal_ratio_b, double *thermal_ratio_c )
{


  if ( (thermal_ratio_b[gridp] > 0.0 && thermal_ratio_c[gridp] > 0.0)
       || (thermal_ratio_b[gridp] < 0.0 && thermal_ratio_c[gridp] < 0.0) ){

    temperature_c[gridp]   = temperature_a[gridp];

    thermal_ratio_c[gridp] = thermal_ratio_a[gridp];

    temperature_d[gridp]   = temperature_b[gridp] - temperature_a[gridp];

    temperature_e[gridp]   = temperature_d[gridp];

  }


  if ( fabs(thermal_ratio_c[gridp]) < fabs(thermal_ratio_b[gridp]) ){

    temperature_a[gridp]   = temperature_b[gridp];
    temperature_b[gridp]   = temperature_c[gridp];
    temperature_c[gridp]   = temperature_a[gridp];

    thermal_ratio_a[gridp] = thermal_ratio_b[gridp];
    thermal_ratio_b[gridp] = thermal_ratio_c[gridp];
    thermal_ratio_c[gridp] = thermal_ratio_a[gridp];

  }

  return(0);

}



int main()
{


  int gridp = 0;

  double temperature_a[NGRID]  = {     0.0 };
  double temperature_b[NGRID]  = { 1000.0 };

  double thermal_ratio_a[NGRID] = { f(temperature_a[0]) };
  double thermal_ratio_b[NGRID] = { f(temperature_b[0]) };

  double temperature_c[NGRID]   = { temperature_b[0] };
  double thermal_ratio_c[NGRID] = { thermal_ratio_b[0] };

  double temperature_d[NGRID];
  double temperature_e[NGRID];

  int niter     = 0;
  int MAX_niter = 100;


  while( niter < MAX_niter ){

    niter++;


    shuffle_x( gridp, temperature_a, temperature_b, temperature_c, temperature_d,
               temperature_e, thermal_ratio_a, thermal_ratio_b, thermal_ratio_c );


    double tolerance = 2.0*EPSILON*fabs(temperature_b[gridp])+0.5*tol1;

    double xm = (temperature_c[gridp] - temperature_b[gridp]) / 2.0;

    if ( (fabs(xm) <= tolerance) || (thermal_ratio_b[gridp] == 0.0) ){


      /* Converged !!! */

      break;

    }

    else {


      update_x( gridp, temperature_a, temperature_b, temperature_c, temperature_d,
                temperature_e, thermal_ratio_a, thermal_ratio_b, thermal_ratio_c );

      printf("iteration %d   temp a %lf   temp b %lf \n", niter, temperature_a[gridp], temperature_b[gridp]);

      thermal_ratio_b[gridp] = f(temperature_b[gridp]);

    }

  }

  double root = temperature_b[gridp];

  printf("The root is %lf \n", root );

  return(0);

}
