/* Frederik De Ceuster - University College London & KU Leuven                                   */
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



#include "../parameters.hpp"
#include "Magritte_config.hpp"
#include "declarations.hpp"



/* level_population_solver: sets up and solves the matrix equation corresp. to equilibrium eq.   */
/*-----------------------------------------------------------------------------------------------*/

int level_population_solver( GRIDPOINT *gridpoint, long gridp, int lspec, double *R, double *pop );

/*-----------------------------------------------------------------------------------------------*/



/* Gauss-Jordan solver for an n by n matrix equation a*x=b and m solution vectors b              */
/*-----------------------------------------------------------------------------------------------*/

int GaussJordan(int n, int m, double *a, double *b);

/*-----------------------------------------------------------------------------------------------*/



#endif /* __LEVEL_POPULATION_SOLVER_HPP_INCLUDED__ */

/*-----------------------------------------------------------------------------------------------*/
