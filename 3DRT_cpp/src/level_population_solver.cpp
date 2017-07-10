/* Frederik De Ceuster - University College London                                               */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/* Solves the equilibrium equation for the level populations                                     */
/*                                                                                               */
/* (based on the Gauss-Jordan solver in Numerical Recipes, Press et al.)                         */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/


#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#define SWAP(a,b) {temp=(a);(a)=(b);(b)=temp;}
#define IND(r,c) ((c)+(r)*n)
#define IMD(r,c) ((c)+(r)*m)



void level_population_solver( double *R, double *pop, double *dpop,
                              double *density, long gridp, int lspec )
{

  int i, j;                                       /* indices for the population level n_i or n_j */

  int n = nlev[lspec];                      /* number of rows and columns of the matrix to solve */
  int m = 1;                                                   /* number of solution vectors 'b' */

  double *a;
  a = (double*) malloc( n*n*sizeof(double) );

  double *b;
  b = (double*) malloc( n*sizeof(double) );

  double out;


  /* Fill the matrix a */  

  for (i=0; i<nlev[lspec]; i++){

    out = 0.0;

    for (j=0; j<nlev[lspec]; j++){

      out = out + R[SPECGRIDLEVLEV(lspec,gridp,i,j)];

      a[LINDEX(i,j)] = R[SPECGRIDLEVLEV(lspec,gridp,j,i)];
    }

    a[LINDEX(i,i)] = -out;
  }



  for (i=0; i<nlev[lspec]; i++){

    b[i] = 0.0;
    a[LINDEX(nlev[lspec]-1, i)] = 1.0E-8;

    dpop[SPECGRIDLEV(lspec,gridp,i)] = pop[SPECGRIDLEV(lspec,gridp,i)];
  }

  b[nlev[lspec]-1] = 1.0E-8 * density[gridp];

  

/*
  printf("Input for lspec %d:", lspec);
  printf("\n");

  for (i=0; i<n; i++){

    for (j=0; j<n; j++){

      printf("%.2lE\t", a[IND(i,j)]);
    }

    printf("\n");
  }

  printf("\n"); */
/*
  for (i=0; i<n; i++){

    for (j=0; j<m; j++){

      printf("%.2lE\t", b[IMD(i,j)]);
    }

    printf("\n");
  }
*/

  /* Solve the system of equations using the Gauss-Jordan solver */

  void GaussJordan(int n, int m, double *a, double *b);

  GaussJordan(n, m, a, b);


  /* Update the populations and the change in populations */

  for (i=0; i<nlev[lspec]; i++){

    pop[SPECGRIDLEV(lspec,gridp,i)] =  b[i];

    dpop[SPECGRIDLEV(lspec,gridp,i)] = fabs( dpop[SPECGRIDLEV(lspec,gridp,i)]
                                             - pop[SPECGRIDLEV(lspec,gridp,i)] );

    if( isnan(b[i]) ){

      printf( "(level_population_solver): population of level (%d,%d) is NaN at grid point %ld \n",
              lspec, i, gridp );
    }

  }

/*
  printf( "(level_population_solver): dpop is %.2lE \n", dpop[SPECGRIDLEV(lspec,gridp,i)] );
*/

/*
  printf("Output:");
  printf("\n");

  for (i=0; i<n; i++){

    for (j=0; j<n; j++){

      printf("%.2lE\t", a[IND(i,j)]);
    }

    printf("\n");
  }

  printf("\n");

  for (i=0; i<n; i++){

    for (j=0; j<m; j++){

        printf("%.2lE\t", b[i]);
    }

    printf("\n");
  }
*/

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

  int i, icol, irow, j, k, l, ll;
  
  double big, dum, pivinv, temp;


  for (j=0; j<n; j++){ ipiv[j] = 0; }
 

  for (i=0; i<n; i++){

    big = 0.0;
    
    for (j=0; j<n; j++){
    
      if (ipiv[j] != 1){
      
        for (k=0; k<n; k++){
        
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

      for (l=0; l<n; l++){ SWAP(a[IND(irow,l)], a[IND(icol,l)]) }
      for (l=0; l<m; l++){ SWAP(b[IMD(irow,l)], b[IMD(icol,l)]) }
    }

    indexr[i] = irow;
    indexc[i] = icol;

    if (a[IND(icol,icol)] == 0.0){ printf("(GaussJordan): ERROR - singular matrix !!!\n"); }

    pivinv = 1.0 / a[IND(icol,icol)];

    a[IND(icol,icol)] = 1.0;

    for (l=0; l<n; l++){ a[IND(icol,l)] = pivinv * a[IND(icol,l)]; }
    for (l=0; l<m; l++){ b[IMD(icol,l)] = pivinv * b[IMD(icol,l)]; }


    for (ll=0; ll<n; ll++){

      if (ll != icol) {

        dum = a[IND(ll,icol)];
        a[IND(ll,icol)] = 0.0;

        for (l=0; l<n; l++){ a[IND(ll,l)] = a[IND(ll,l)] - a[IND(icol,l)]*dum; }
        for (l=0; l<m; l++){ b[IMD(ll,l)] = b[IMD(ll,l)] - b[IMD(icol,l)]*dum; }
      }

    }

  }


  for (l=n-1; l>=0; l--){

    if (indexr[l] != indexc[l] ){

      for (k=0; k<n; k++){ SWAP(a[IND(k,indexr[l])], a[IND(k,indexc[l])]); }   
   
    }

  }

}

/*-----------------------------------------------------------------------------------------------*/
