/* Frederik De Ceuster - University College London                                               */
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



/* Private functions to output results */
/*-----------------------------------------------------------------------------------------------*/

static void PrintOutput(realtype t, realtype y1, realtype y2, realtype y3);
static void PrintRootInfo(int root_f1, int root_f2);

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

static int Jac(long int N, realtype t,
               N_Vector y, N_Vector fy, DlsMat J, void *user_data,
               N_Vector tmp1, N_Vector tmp2, N_Vector tmp3);

/*-----------------------------------------------------------------------------------------------*/



#endif /* __RATE_EQUATION_SOLVER_H_INCLUDED__ */

/*-----------------------------------------------------------------------------------------------*/
