// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __LEVEL_POPULATION_SOLVER_OTF_HPP_INCLUDED__
#define __LEVEL_POPULATION_SOLVER_OTF_HPP_INCLUDED__


#include "declarations.hpp"


// level_population_solver: sets up and solves matrix equation corresp. to equilibrium eq.
// ---------------------------------------------------------------------------------------

int level_population_solver_otf (CELL *cell, long gridp, int lspec, double *R, double *pop);


// Gauss-Jordan solver for an n by n matrix equation a*x=b and m solution vectors b
// --------------------------------------------------------------------------------

int GaussJordan (int n, int m, double *a, double *b);


#endif // __LEVEL_POPULATION_SOLVER_OTF_HPP_INCLUDED__
