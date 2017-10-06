/* Frederik De Ceuster - University College London & KU Leuven                                   */
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

#include <Eigen/Dense>
using namespace Eigen;

#include "declarations.hpp"
#include "level_population_solver.hpp"



/* level_population_solver: sets up and solves the matrix equation corresp. to equilibrium eq.   */
/*-----------------------------------------------------------------------------------------------*/

int level_population_solver( GRIDPOINT *gridpoint, long gridp, int lspec, double *R,
                             double *pop, double *dpop )
{


  MatrixXd A(nlev[lspec],nlev[lspec]);

  VectorXd b(nlev[lspec]);





  /*   Fill matrix a and vector b                                                                */
  /*_____________________________________________________________________________________________*/


  for (int i=0; i<nlev[lspec]; i++){

    double row_tot = 0.0;

    for (int j=0; j<nlev[lspec]; j++){

      row_tot = row_tot + R[LSPECGRIDLEVLEV(lspec,gridp,i,j)];

      A(i,j) = R[LSPECGRIDLEVLEV(lspec,gridp,j,i)];
    }

    A(i,i) = -row_tot;
  }



  for (int i=0; i<nlev[lspec]; i++){

    b(i) = 0.0;

    A(nlev[lspec]-1, i) = 1.0;

    dpop[LSPECGRIDLEV(lspec,gridp,i)] = pop[LSPECGRIDLEV(lspec,gridp,i)];
  }


  b(nlev[lspec]-1) = gridpoint[gridp].density;


  /*_____________________________________________________________________________________________*/





  /* Solve the system of equations using the colPivHouseholderQr solver provided by Eigen */

  VectorXd x = A.colPivHouseholderQr().solve(b);





  /* UPDATE THE POPULATIONS AND THE CHANGE IN POPULATIONS                                        */
  /*_____________________________________________________________________________________________*/


  for (int i=0; i<nlev[lspec]; i++){

    long p_i = LSPECGRIDLEV(lspec,gridp,i);


    /* avoid too small or too large populations */

    if (x(i) > POP_LOWER_LIMIT){

      if ( x(i) < POP_UPPER_LIMIT ) { pop[p_i] =  x(i); }

      else                          { pop[p_i] = POP_UPPER_LIMIT; }
    }
    else {

      pop[p_i] = 0.0;
    }


    dpop[p_i] = fabs(dpop[p_i] - pop[p_i]);


    if( isnan(b(i)) ){

      printf( "\n\n !!! ERROR in level poopulation solver !!!\n\n" );
      printf( "   [ERROR]: population (%d,%d) is NaN at grid point %ld \n", lspec, i, gridp );
    }

  }


  /*_____________________________________________________________________________________________*/





  return(0);

}

/*-----------------------------------------------------------------------------------------------*/
