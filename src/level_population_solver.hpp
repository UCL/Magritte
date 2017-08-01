/* Frederik De Ceuster - University College London                                               */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/* Header for level_population_solver.cpp                                                        */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/



#ifndef __LEVEL_POPULATION_SOLVER_HPP_INCLUDED__
#define __LEVEL_POPULATION_SOLVER_HPP_INCLUDED__



#include "declarations.hpp"



/* level_population_solver: sets up and solves the matrix equation corresp. to equilibrium eq.   */
/*-----------------------------------------------------------------------------------------------*/

void level_population_solver( GRIDPOINT *gridpoint, double *R, double *pop, double *dpop,
                              long gridp, int lspec );

/*-----------------------------------------------------------------------------------------------*/



/* Gauss-Jordan solver for an n by n matrix equation a*x=b and m solution vectors b              */
/*-----------------------------------------------------------------------------------------------*/

void GaussJordan(int n, int m, double *a, double *b);

/*-----------------------------------------------------------------------------------------------*/



#endif /* __LEVEL_POPULATION_SOLVER_HPP_INCLUDED__ */

/*-----------------------------------------------------------------------------------------------*/
