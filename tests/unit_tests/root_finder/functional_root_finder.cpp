
#include <math.h>
#include <stdio.h>



#define EPSILON 3.0E-8
#define PREC    5.0E-3
#define tol1    PREC
#define NCELLS   1


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



int update_x( long o, double *temperature_a, double *temperature_b, double *temperature_c,
              double *temperature_d, double *temperature_e,
              double *thermal_ratio_a, double *thermal_ratio_b, double *thermal_ratio_c )
{


  double tolerance = 2.0*EPSILON*fabs(temperature_b[o])+0.5*tol1;

  double xm = (temperature_c[o] - temperature_b[o]) / 2.0;


  if ( (fabs(temperature_e[o]) >= tolerance)
       && (fabs(thermal_ratio_a[o]) > fabs(thermal_ratio_b[o])) ){


    /* Attempt inverse quadratic interpolation */

    double s = thermal_ratio_b[o] / thermal_ratio_a[o];

    double p, q, r;

    if (temperature_a[o] == temperature_c[o]){

      p = 2.0 * xm * s;
      q = 1.0 - s;
    }

    else {

      q = thermal_ratio_a[o] / thermal_ratio_c[o];
      r = thermal_ratio_b[o] / thermal_ratio_c[o];
      p = s*( 2.0*xm*q*(q-r) - (temperature_b[o]-temperature_a[o])*(r-1.0) );
      q = (q-1.0)*(r-1.0)*(s-1.0);
    }

    if (p > 0.0){ q = -q; }

    p = fabs(p);


    double min1 = 3.0*xm*q - fabs(tolerance*q);
    double min2 = fabs(temperature_e[o]*q);


    if (2.0*p < (min1 < min2 ? min1 :  min2)){  /* Accept interpolation */

      temperature_e[o] = temperature_d[o];
      temperature_d[o] = p / q;
    }

    else {  /* Interpolation failed, use bisection */

      temperature_d[o] = xm;
      temperature_e[o] = temperature_d[o];
    }

  }

  else {  /* Bounds decreasing too slowly, use bisection */

    temperature_d[o] = xm;
    temperature_e[o] = temperature_d[o];
  }


  /* Move last best guess to temperature_a */

  temperature_a[o]   = temperature_b[o];

  thermal_ratio_a[o] = thermal_ratio_b[o];


  /* Evaluate new trial root */

  if ( fabs(temperature_d[o]) > tolerance ){

    temperature_b[o] = temperature_b[o] + temperature_d[o];
  }

  else {

    if (xm > 0.0){ temperature_b[o] = temperature_b[o] + fabs(tolerance); }

    else { temperature_b[o] = temperature_b[o] - fabs(tolerance); }
  }


  return(0);

}

int shuffle_x( long o, double *temperature_a, double *temperature_b, double *temperature_c,
               double *temperature_d, double *temperature_e,
               double *thermal_ratio_a, double *thermal_ratio_b, double *thermal_ratio_c )
{


  if ( (thermal_ratio_b[o] > 0.0 && thermal_ratio_c[o] > 0.0)
       || (thermal_ratio_b[o] < 0.0 && thermal_ratio_c[o] < 0.0) ){

    temperature_c[o]   = temperature_a[o];

    thermal_ratio_c[o] = thermal_ratio_a[o];

    temperature_d[o]   = temperature_b[o] - temperature_a[o];

    temperature_e[o]   = temperature_d[o];

  }


  if ( fabs(thermal_ratio_c[o]) < fabs(thermal_ratio_b[o]) ){

    temperature_a[o]   = temperature_b[o];
    temperature_b[o]   = temperature_c[o];
    temperature_c[o]   = temperature_a[o];

    thermal_ratio_a[o] = thermal_ratio_b[o];
    thermal_ratio_b[o] = thermal_ratio_c[o];
    thermal_ratio_c[o] = thermal_ratio_a[o];

  }

  return(0);

}



int main()
{


  int o = 0;

  double temperature_a[NCELLS]  = {     0.0 };
  double temperature_b[NCELLS]  = { 1000.0 };

  double thermal_ratio_a[NCELLS] = { f(temperature_a[0]) };
  double thermal_ratio_b[NCELLS] = { f(temperature_b[0]) };

  double temperature_c[NCELLS]   = { temperature_b[0] };
  double thermal_ratio_c[NCELLS] = { thermal_ratio_b[0] };

  double temperature_d[NCELLS];
  double temperature_e[NCELLS];

  int niter     = 0;
  int MAX_niter = 100;


  while( niter < MAX_niter ){

    niter++;


    shuffle_x( o, temperature_a, temperature_b, temperature_c, temperature_d,
               temperature_e, thermal_ratio_a, thermal_ratio_b, thermal_ratio_c );


    double tolerance = 2.0*EPSILON*fabs(temperature_b[o])+0.5*tol1;

    double xm = (temperature_c[o] - temperature_b[o]) / 2.0;

    if ( (fabs(xm) <= tolerance) || (thermal_ratio_b[o] == 0.0) ){


      /* Converged !!! */

      break;

    }

    else {


      update_x( o, temperature_a, temperature_b, temperature_c, temperature_d,
                temperature_e, thermal_ratio_a, thermal_ratio_b, thermal_ratio_c );

      printf("iteration %d   temp a %lf   temp b %lf \n", niter, temperature_a[o], temperature_b[o]);

      thermal_ratio_b[o] = f(temperature_b[o]);

    }

  }

  double root = temperature_b[o];

  printf("The root is %lf \n", root );

  return(0);

}
