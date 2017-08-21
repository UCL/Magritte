/* Frederik De Ceuster - University College London                                               */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/* level_population_solver: Solves the equilibrium equation for the level populations            */
/*                                                                                               */
/* (based on the Gauss-Jordan solver in Numerical Recipes, Press et al.)                         */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/


#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#include "level_population_solver.hpp"


#define SWAP(a,b) {temp=(a);(a)=(b);(b)=temp;}
#define IND(r,c) ((c)+(r)*n)
#define IMD(r,c) ((c)+(r)*m)



void level_population_solver( GRIDPOINT *gridpoint, long gridp, int lspec, double *R,
                              double *pop, double *dpop )
{


  int n = nlev[lspec];                      /* number of rows and columns of the matrix to solve */
  int m = 1;                                                   /* number of solution vectors 'b' */

  double *a;
  a = (double*) malloc( nlev[lspec]*nlev[lspec]*sizeof(double) );

  double *b;
  b = (double*) malloc( nlev[lspec]*sizeof(double) );




  /* Fill the matrix a */

  for (int i=0; i<nlev[lspec]; i++){

    double out = 0.0;

    for (int j=0; j<nlev[lspec]; j++){

      out = out + R[LSPECGRIDLEVLEV(lspec,gridp,i,j)];

      a[LINDEX(i,j)] = R[LSPECGRIDLEVLEV(lspec,gridp,j,i)];
    }

    a[LINDEX(i,i)] = -out;
  }



  for (int i=0; i<nlev[lspec]; i++){

    b[i] = 0.0;

    a[LINDEX(nlev[lspec]-1, i)] = 1.0E-8;

    dpop[LSPECGRIDLEV(lspec,gridp,i)] = pop[LSPECGRIDLEV(lspec,gridp,i)];
  }

  b[nlev[lspec]-1] = 1.0E-8 * gridpoint[gridp].density;



  /* Solve the system of equations using the Gauss-Jordan solver */

  GaussJordan(n, m, a, b);


  /* Update the populations and the change in populations */

  for (int i=0; i<nlev[lspec]; i++){

    pop[LSPECGRIDLEV(lspec,gridp,i)] =  b[i];

    dpop[LSPECGRIDLEV(lspec,gridp,i)] = fabs( dpop[LSPECGRIDLEV(lspec,gridp,i)]
                                             - pop[LSPECGRIDLEV(lspec,gridp,i)] );

    if( isnan(b[i]) ){

      printf( "(level_population_solver): population of level (%d,%d) is NaN at grid point %ld \n",
              lspec, i, gridp );
    }

  }



  /* Free the allocated memory for temporary variables */

  free(a);
  free(b);

}




/* Gauss-Jordan solver for an n by n matrix equation a*x=b and m solution vectors b              */
/*-----------------------------------------------------------------------------------------------*/

void GaussJordan(int n, int m, double *a, double *b)
{

  int indexc[n];                              /* note that our vectors are indexed from 0 to n-1 */
  int indexr[n];                              /* note that our vectors are indexed from 0 to n-1 */
  int ipiv[n];

  int icol, irow;

  double temp;


  for (int j=0; j<n; j++){ ipiv[j] = 0; }


  for (int i=0; i<n; i++){

    double big = 0.0;

    for (int j=0; j<n; j++){

      if (ipiv[j] != 1){

        for (int k=0; k<n; k++){

          if (ipiv[k] == 0){

            if(fabs(a[IND(j,k)]) >= big){

              big = fabs(a[IND(j,k)]);
              irow = j;
              icol = k;
            }

          }

        }

      }

    }


    ipiv[icol] = ipiv[icol] + 1;

    if (irow != icol){

      for (int l=0; l<n; l++){ SWAP(a[IND(irow,l)], a[IND(icol,l)]) }
      for (int l=0; l<m; l++){ SWAP(b[IMD(irow,l)], b[IMD(icol,l)]) }
    }

    indexr[i] = irow;
    indexc[i] = icol;

    if (a[IND(icol,icol)] == 0.0){ printf("(GaussJordan): ERROR - singular matrix !!!\n"); }

    double pivinv = 1.0 / a[IND(icol,icol)];

    a[IND(icol,icol)] = 1.0;

    for (int l=0; l<n; l++){ a[IND(icol,l)] = pivinv * a[IND(icol,l)]; }
    for (int l=0; l<m; l++){ b[IMD(icol,l)] = pivinv * b[IMD(icol,l)]; }


    for (int ll=0; ll<n; ll++){

      if (ll != icol) {

        double dum = a[IND(ll,icol)];

        a[IND(ll,icol)] = 0.0;

        for (int l=0; l<n; l++){ a[IND(ll,l)] = a[IND(ll,l)] - a[IND(icol,l)]*dum; }
        for (int l=0; l<m; l++){ b[IMD(ll,l)] = b[IMD(ll,l)] - b[IMD(icol,l)]*dum; }
      }

    }

  }


  for (int l=n-1; l>=0; l--){

    if (indexr[l] != indexc[l] ){

      for (int k=0; k<n; k++){ SWAP(a[IND(k,indexr[l])], a[IND(k,indexc[l])]); }

    }

  }

}

/*-----------------------------------------------------------------------------------------------*/
