// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#include "../parameters.hpp"
#include "Magritte_config.hpp"
#include "declarations.hpp"

#include "level_population_solver.hpp"


#define SWAP(a,b) {temp=(a);(a)=(b);(b)=temp;}
#define IND(r,c) ((c)+(r)*n)
#define IMD(r,c) ((c)+(r)*m)


int level_population_solver (long ncells, CELL *cell, LINE_SPECIES line_species,
                             long gridp, int lspec, double *R, double *pop)
{

  const int n = nlev[lspec];   // number of rows and columns of matrix
  const int m = 1;             // number of solution vectors b

  double *a = new double[n*n];
  double *b = new double[n*m];




  // Fill matrix a and vector b
  // __________________________


  for (int i = 0; i < n; i++)
  {
    double out = 0.0;

    for (int j = 0; j < n; j++)
    {
      out = out + R[LSPECLEVLEV(lspec,i,j)];

      a[LINDEX(i,j)] = R[LSPECLEVLEV(lspec,j,i)];
    }

    a[LINDEX(i,i)] = -out;
  }


  for (int i = 0; i < n; i++)
  {
    b[i] = 0.0;

    a[LINDEX(n-1, i)] = 1.0;
  }


  b[nlev[lspec]-1] = cell[gridp].density * cell[gridp].abundance[line_species.nr[lspec]];




  // Solve system of equations using Gauss-Jordan solver
  // ___________________________________________________


  GaussJordan(n, m, a, b);




  // UPDATE POPULATIONS AND CHANGE IN POPULATIONS
  // ____________________________________________


  for (int i = 0; i < nlev[lspec]; i++)
  {
    long p_i = LSPECGRIDLEV(lspec,gridp,i);


    // avoid too small or too large populations

    if (b[i] > POP_LOWER_LIMIT)
    {
      if (b[i] < POP_UPPER_LIMIT)
      {
        pop[p_i] =  b[i];
      }

      else
      {
        pop[p_i] = POP_UPPER_LIMIT;
      }
    }

    else
    {
      pop[p_i] = 0.0;
    }

  }


  delete [] a;
  delete [] b;


  return(0);

}




// Gauss-Jordan solver for an n by n matrix equation a*x=b and m solution vectors b
// --------------------------------------------------------------------------------

int GaussJordan (int n, int m, double *a, double *b)
{

  // based on the Gauss-Jordan solver in Numerical Recipes, Press et al.

  int *indexc = new int[n];   // note that our vectors are indexed from 0 to n-1
  int *indexr = new int[n];   // note that our vectors are indexed from 0 to n-1

  int *ipiv = new int[n];


  int icol, irow;

  double temp;


  for (int j = 0; j < n; j++)
  {
    ipiv[j] = 0;
  }


  for (int i = 0; i < n; i++)
  {
    double big = 0.0;

    for (int j = 0; j < n; j++)
    {
      if (ipiv[j] != 1)
      {
        for (int k = 0; k < n; k++)
        {
          if (ipiv[k] == 0)
          {
            if (fabs(a[IND(j,k)]) >= big)
            {
              big = fabs(a[IND(j,k)]);
              irow = j;
              icol = k;
            }
          }
        }
      }
    }


    ipiv[icol] = ipiv[icol] + 1;

    if (irow != icol)
    {
      for (int l = 0; l < n; l++)
      {
        SWAP(a[IND(irow,l)], a[IND(icol,l)]);
      }

      for (int l = 0; l < m; l++)
      {
        SWAP(b[IMD(irow,l)], b[IMD(icol,l)]);
      }
    }

    indexr[i] = irow;
    indexc[i] = icol;

    if (a[IND(icol,icol)] == 0.0)
    {
      printf("(GaussJordan): ERROR - singular matrix !!!\n");
    }

    double pivinv = 1.0 / a[IND(icol,icol)];

    a[IND(icol,icol)] = 1.0;

    for (int l = 0; l < n; l++)
    {
      a[IND(icol,l)] = pivinv * a[IND(icol,l)];
    }

    for (int l = 0; l < m; l++)
    {
      b[IMD(icol,l)] = pivinv * b[IMD(icol,l)];
    }


    for (int ll = 0; ll < n; ll++)
    {
      if (ll != icol)
      {
        double dum = a[IND(ll,icol)];

        a[IND(ll,icol)] = 0.0;

        for (int l = 0; l < n; l++)
        {
          a[IND(ll,l)] = a[IND(ll,l)] - a[IND(icol,l)]*dum;
        }

        for (int l = 0; l < m; l++)
        {
          b[IMD(ll,l)] = b[IMD(ll,l)] - b[IMD(icol,l)]*dum;
        }
      }
    }

  } // end of i loop over rows


  for (int l = n-1; l >= 0; l--)
  {
    if (indexr[l] != indexc[l] )
    {
      for (int k = 0; k < n; k++)
      {
        SWAP(a[IND(k,indexr[l])], a[IND(k,indexc[l])]);
      }
    }
  }


  delete [] indexc;
  delete [] indexr;
  delete [] ipiv;


  return(0);

}
