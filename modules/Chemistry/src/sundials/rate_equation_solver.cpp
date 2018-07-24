// Magritte: Multidimensional Accelerated General-purpose Radiative Transfer
//
// Developed by: Frederik De Ceuster - University College London & KU Leuven
// _________________________________________________________________________


#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <cvode/cvode.h>                 // prototypes for CVODE fcts., consts.
#include <nvector/nvector_serial.h>      // serial N_Vector types, fcts., macros
#include <sunmatrix/sunmatrix_dense.h>   // access to dense SUNMatrix
#include <sunlinsol/sunlinsol_dense.h>   // access to dense SUNLinearSolver
#include <cvode/cvode_direct.h>          // access to CVDls interface
#include <sundials/sundials_types.h>     // definition of type realtype

#include "../declarations.hpp"
#include "rate_equation_solver.hpp"
#include "rate_equations.cpp"
// #include "jacobian.cpp"


// User-defined vector and matrix accessor macros: Ith, IJth

#define Ith(v,i)    NV_Ith_S(v,i)         // Ith numbers components 0..NEQ-1
#define IJth(A,i,j) SM_ELEMENT_D(A,i,j)   // IJth numbers rows,cols 0..NEQ-1


// Problem Constants

#define NEQ      (NSPEC-3)         // number of equations: NSPEC minus dummies minus electrons
#define RTOL     RCONST(1.0E-8)    // scalar relative tolerance
#define ATOL     RCONST(1.0e-30)   // vector absolute tolerance components



// rate_equation_solver: solves rate equations given in rate_equations.cpp
// -----------------------------------------------------------------------

int rate_equation_solver (CELLS *cells, SPECIES species, long o)
{


  // Prepare data to be passed to solver routines

  USER_DATA user_data;

  user_data = NULL;
  user_data = (USER_DATA) malloc( sizeof(*user_data) );

  user_data->gp          = o;
  user_data->cellpointer = cells;
  user_data->electron_abundance = cells->abundance[SINDEX(o,species.nr_e)];


  SUNMatrix       A  = NULL;
  SUNLinearSolver LS = NULL;


  int flag;     // output flag for CVODE functions

  realtype t;   // output time for solver


  realtype time_start = 0.0;                                 // start time of chemical evolution

  realtype time_end   = TIME_END_IN_YEARS*SECONDS_IN_YEAR;   // end time of chemical evolution


  // Specify maximum number of internal steps

  int mxstep = 1000000;


  // Create serial vector of length NEQ for I.C. and abstol

  N_Vector y = N_VNew_Serial(NEQ);

  if (check_flag((void *)y, "N_VNew_Serial", 0))
  {
    return (1);
  }


  N_Vector abstol = N_VNew_Serial(NEQ);

  if (check_flag((void *)abstol, "N_VNew_Serial", 0))
  {
    return (1);
  }


  // Initialize y

  for (int i = 0; i < NEQ; i++)
  {
    if (cells->abundance[SINDEX(o,i+1)] > 0.0)
    {
      Ith(y,i) = (realtype) cells->abundance[SINDEX(o,i+1)];
    }

    else
    {
      Ith(y,i) = 0.0;
    }
  }


  // Set scalar relative tolerance

  realtype reltol = RTOL;


  // Set vector absolute tolerance

  for (int i = 0; i < NEQ; i++)
  {
    Ith(abstol,i) = ATOL;
  }



  /* Call CVodeCreate to create solver memory and specify
   * Backward Differentiation Formula and use of a Newton iteration */

  void *cvode_mem = NULL;

  cvode_mem = CVodeCreate(CV_BDF, CV_NEWTON);

  if (check_flag((void *)cvode_mem, "CVodeCreate", 0))
  {
    return (1);
  }


  /* Call CVodeInit to initialize integrator memory and specify
   * user's right hand side function in y'=f(t,y), the inital time time_start, and
   * initial dependent variable vector y. */

  flag = CVodeInit(cvode_mem, f, time_start, y);

  if (check_flag(&flag, "CVodeInit", 1))
  {
    return (1);
  }


  /* Call CVodeSVtolerances to specify scalar relative tolerance
   * and vector absolute tolerances */

  flag = CVodeSVtolerances(cvode_mem, reltol, abstol);

  if (check_flag(&flag, "CVodeSVtolerances", 1))
  {
    return (1);
  }


  // Create dense SUNMatrix for use in linear solves

  A = SUNDenseMatrix(NEQ, NEQ);

  if (check_flag((void *)A, "SUNDenseMatrix", 0))
  {
    return (1);
  }


  // Create dense SUNLinearSolver object for use by CVode

  LS = SUNDenseLinearSolver(y, A);

  if (check_flag((void *)LS, "SUNDenseLinearSolver", 0))
  {
    return (1);
  }


  // Call CVDlsSetLinearSolver to attach matrix and linear solver to CVode

  flag = CVDlsSetLinearSolver(cvode_mem, LS, A);

  if (check_flag(&flag, "CVDlsSetLinearSolver", 1))
  {
    return (1);
  }


  // Specify that a user-supplied Jacobian function (Jac) is to be used
  //
  // flag = CVDlsSetJacFn(cvode_mem, Jac);
  //
  // if (check_flag(&flag, "CVDlsSetJacFn", 1))
  // {
  //   return (1);
  // }


  // Call CVodeSetMaxNumSteps to set maximum number of steps

  flag = CVodeSetMaxNumSteps(cvode_mem, mxstep);

  if (check_flag(&flag, "CVodeSetMaxNumSteps", 1))
  {
    return (1);
  }


  // Specify user-defined data to be passed to various routines

  flag = CVodeSetUserData(cvode_mem, user_data);

  if (check_flag(&flag, "CVodeSetUserData", 1))
  {
    return (1);
  }


  // Call CVode

  flag = CVode(cvode_mem, time_end, y, &t, CV_NORMAL);

  if (check_flag(&flag, "CVode", 1))
  {
    printf("\n\n !!! CVode ERROR in solver !!! \n\n");

    return (1);
  }


  // Update abundances for each species

  for (int i = 0; i < NEQ; i++)
  {
    if (Ith(y,i) > 1.0E-30)
    {
      cells->abundance[SINDEX(o,i+1)] = Ith(y,i);
    }

    else
    {
      cells->abundance[SINDEX(o,i+1)] = 0.0;
    }
  }

  cells->abundance[SINDEX(o,species.nr_e)] = user_data->electron_abundance;


  // Print some final statistics

  // PrintFinalStats(cvode_mem);


  // Free y and abstol vectors

  N_VDestroy_Serial(y);
  N_VDestroy_Serial(abstol);


  // Free integrator memory

  CVodeFree(&cvode_mem);


  // Free linear solver memory

  SUNLinSolFree(LS);


  // Free matrix memory

  SUNMatDestroy(A);


  // Free user data

  // free(user_data);


  return (0);

}




// Private helper functions
// ________________________




// PrintFinalStats: Get and print some final statistics
// ----------------------------------------------------

static void PrintFinalStats (void *cvode_mem)
{


  long int nst, nfe, nsetups, nje, nfeLS, nni, ncfn, netf, nge;
  int flag;

  flag = CVodeGetNumSteps(cvode_mem, &nst);
  check_flag(&flag, "CVodeGetNumSteps", 1);
  flag = CVodeGetNumRhsEvals(cvode_mem, &nfe);
  check_flag(&flag, "CVodeGetNumRhsEvals", 1);
  flag = CVodeGetNumLinSolvSetups(cvode_mem, &nsetups);
  check_flag(&flag, "CVodeGetNumLinSolvSetups", 1);
  flag = CVodeGetNumErrTestFails(cvode_mem, &netf);
  check_flag(&flag, "CVodeGetNumErrTestFails", 1);
  flag = CVodeGetNumNonlinSolvIters(cvode_mem, &nni);
  check_flag(&flag, "CVodeGetNumNonlinSolvIters", 1);
  flag = CVodeGetNumNonlinSolvConvFails(cvode_mem, &ncfn);
  check_flag(&flag, "CVodeGetNumNonlinSolvConvFails", 1);

  flag = CVDlsGetNumJacEvals(cvode_mem, &nje);
  check_flag(&flag, "CVDlsGetNumJacEvals", 1);
  flag = CVDlsGetNumRhsEvals(cvode_mem, &nfeLS);
  check_flag(&flag, "CVDlsGetNumRhsEvals", 1);

  flag = CVodeGetNumGEvals(cvode_mem, &nge);
  check_flag(&flag, "CVodeGetNumGEvals", 1);

  printf("\nFinal Statistics:\n");
  printf("nst = %-6ld nfe  = %-6ld nsetups = %-6ld nfeLS = %-6ld nje = %ld\n",
	        nst, nfe, nsetups, nfeLS, nje);
  printf("nni = %-6ld ncfn = %-6ld netf = %-6ld nge = %ld\n \n",
	       nni, ncfn, netf, nge);
}




// check_flag
// ----------

static int check_flag (void *flagvalue, const char *funcname, int opt)
{

/*
 * Check function return value...
 *   opt == 0 means SUNDIALS function allocates memory so check if
 *            returned NULL pointer
 *   opt == 1 means SUNDIALS function returns a flag so check if
 *            flag >= 0
 *   opt == 2 means function allocates memory so check if returned
 *            NULL pointer
 */

  int *errflag;


  // Check if SUNDIALS function returned NULL pointer - no memory allocated

  if ( (opt == 0) && (flagvalue == NULL))
  {
    fprintf(stderr, "\nSUNDIALS_ERROR: %s() failed - returned NULL pointer\n\n",
	           funcname);

    return (1);
  }


  // Check if flag < 0

  else if (opt == 1)
  {
    errflag = (int *) flagvalue;

    if (*errflag < 0)
    {
      fprintf (stderr, "\nSUNDIALS_ERROR: %s() failed with flag = %d\n\n",
	             funcname, *errflag);

      return (1);
    }
  }


  // Check if function returned NULL pointer - no memory allocated

  else if ( (opt == 2) && (flagvalue == NULL) )
  {
    fprintf (stderr, "\nMEMORY_ERROR: %s() failed - returned NULL pointer\n\n",
	           funcname);

    return (1);
  }

  return (0);
}