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


#include <cvode/cvode.h>                                  /* prototypes for CVODE fcts., consts. */
#include <nvector/nvector_serial.h>                      /* serial N_Vector types, fcts., macros */

#include <sundials/sundials_types.h>                              /* definition of type realtype */


typedef struct
{
  long gp;

  CELL *cellpointer;

  double electron_abundance;

} *USER_DATA;


/* rate_equation_solver: solves the rate equations given in rate_equations.s                     */
/*-----------------------------------------------------------------------------------------------*/

int rate_equation_solver (CELL *cell, long gridp);

/*-----------------------------------------------------------------------------------------------*/



/* Private function to print final statistics */
/*-----------------------------------------------------------------------------------------------*/

static void PrintFinalStats (void *cvode_mem);

/*-----------------------------------------------------------------------------------------------*/



/* Private function to check function return values */
/*-----------------------------------------------------------------------------------------------*/

static int check_flag (void *flagvalue, const char *funcname, int opt);

/*-----------------------------------------------------------------------------------------------*/



/* Functions Called by the Solver */
/*-----------------------------------------------------------------------------------------------*/

static int f (realtype t, N_Vector y, N_Vector ydot, void *user_data);

/*-----------------------------------------------------------------------------------------------*/



#endif /* __RATE_EQUATION_SOLVER_H_INCLUDED__ */

/*-----------------------------------------------------------------------------------------------*/
