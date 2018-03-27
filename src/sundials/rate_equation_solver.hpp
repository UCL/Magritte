// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#ifndef __RATE_EQUATION_SOLVER_H_INCLUDED__
#define __RATE_EQUATION_SOLVER_H_INCLUDED__


#include <cvode/cvode.h>               // prototypes for CVODE fcts., consts.
#include <nvector/nvector_serial.h>    // serial N_Vector types, fcts., macros
#include <sundials/sundials_types.h>   // definition of type realtype


typedef struct
{
  long gp;

  CELLS *cellpointer;

  double electron_abundance;

} *USER_DATA;


// rate_equation_solver: solves rate equations given in rate_equations.cpp
// -----------------------------------------------------------------------

int rate_equation_solver (CELLS *cells, SPECIES species, long o);


// Private function to print final statistics
// ------------------------------------------

static void PrintFinalStats (void *cvode_mem);


// Private function to check function return values
// ------------------------------------------------

static int check_flag (void *flagvalue, const char *funcname, int opt);


// Functions Called by Solver
// --------------------------

static int f (realtype t, N_Vector y, N_Vector ydot, void *user_data);


#endif // __RATE_EQUATION_SOLVER_H_INCLUDED__
