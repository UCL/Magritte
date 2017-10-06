/* Frederik De Ceuster - University College London & KU Leuven                                   */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/* Header for rate_equation_solver.c                                                             */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/



#ifndef __RATE_EQUATION_SOLVER_H_INCLUDED__
#define __RATE_EQUATION_SOLVER_H_INCLUDED__

#include <string>
#include <iostream>
using namespace std;

#include <cvode/cvode.h>                                  /* prototypes for CVODE fcts., consts. */
#include <nvector/nvector_serial.h>                      /* serial N_Vector types, fcts., macros */
#include <cvode/cvode_dense.h>                                          /* prototype for CVDense */
#include <sundials/sundials_dense.h>                            /* definitions DlsMat DENSE_ELEM */
#include <sundials/sundials_types.h>                              /* definition of type realtype */



typedef struct {

  long gp;
  GRIDPOINT* gridpointer;

} *USER_DATA;



/* rate_equation_solver: solves the rate equations given in rate_equations.s                     */
/*-----------------------------------------------------------------------------------------------*/

int rate_equation_solver(GRIDPOINT *gridpoint, long gridp);

/*-----------------------------------------------------------------------------------------------*/



/* Private function to print final statistics */
/*-----------------------------------------------------------------------------------------------*/

static void PrintFinalStats(void *cvode_mem);

/*-----------------------------------------------------------------------------------------------*/



/* Private function to check function return values */
/*-----------------------------------------------------------------------------------------------*/

static int check_flag(void *flagvalue, const char *funcname, int opt);

/*-----------------------------------------------------------------------------------------------*/



/* Functions Called by the Solver */
/*-----------------------------------------------------------------------------------------------*/

static int f(realtype t, N_Vector y, N_Vector ydot, void *user_data);

/*-----------------------------------------------------------------------------------------------*/



#endif /* __RATE_EQUATION_SOLVER_H_INCLUDED__ */

/*-----------------------------------------------------------------------------------------------*/
