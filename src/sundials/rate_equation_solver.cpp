/* Frederik De Ceuster - University College London                                               */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/* rate_equation_solver: Solver for the system of rate equations                                 */
/*                                                                                               */
/* ( based on calculate_abundances in 3D-PDR                                                     */
/*   and the cvRobers_dns example that comes with Sundials )                                     */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/



#include <stdio.h>

/* Header files with a description of contents used */

#include <cvode/cvode.h>             /* prototypes for CVODE fcts., consts. */
#include <nvector/nvector_serial.h>  /* serial N_Vector types, fcts., macros */
#include <cvode/cvode_dense.h>       /* prototype for CVDense */
#include <sundials/sundials_dense.h> /* definitions DlsMat DENSE_ELEM */
#include <sundials/sundials_types.h> /* definition of type realtype */

#include "../declarations.hpp"
#include "rate_equation_solver.hpp"
#include "jacobian.cpp"
#include "rate_equations.cpp"



/* User-defined vector and matrix accessor macros: Ith, IJth */

/* These macros are defined in order to write code which exactly matches
   the mathematical problem description given above.

   Ith(v,i) references the ith component of the vector v, where i is in
   the range [1..NEQ] and NEQ is defined below. The Ith macro is defined
   using the N_VIth macro in nvector.h. N_VIth numbers the components of
   a vector starting from 0.

   IJth(A,i,j) references the (i,j)th element of the dense matrix A, where
   i and j are in the range [1..NEQ]. The IJth macro is defined using the
   DENSE_ELEM macro in dense.h. DENSE_ELEM numbers rows and columns of a
   dense matrix starting from 0. */

#define Ith(v,i)    NV_Ith_S(v,i-1)       /* Ith numbers components 1..NEQ */
#define IJth(A,i,j) DENSE_ELEM(A,i-1,j-1) /* IJth numbers rows,cols 1..NEQ */


/* Problem Constants */

#define NEQ   NSPEC            /* number of equations  */
// #define Y1    RCONST(1.0)      /* initial y components */
// #define Y2    RCONST(0.0)
// #define Y3    RCONST(0.0)
#define RTOL  RCONST(1.0e-4)   /* scalar relative tolerance            */
#define ATOL  RCONST(1.0e-8)   /* vector absolute tolerance components */
// #define ATOL1 RCONST(1.0e-8)   /* vector absolute tolerance components */
// #define ATOL2 RCONST(1.0e-14)
// #define ATOL3 RCONST(1.0e-6)
#define T0    RCONST(0.0)      /* initial time           */
#define T1    RCONST(0.1)      /* first output time      */
#define TMULT RCONST(10.0)     /* output time factor     */
#define NOUT  1                /* number of output times */



/* Functions Called by the Solver */

static int f(realtype t, N_Vector y, N_Vector ydot, void *user_data);

static int Jac(long int N, realtype t,
               N_Vector y, N_Vector fy, DlsMat J, void *user_data,
               N_Vector tmp1, N_Vector tmp2, N_Vector tmp3);


/* Private functions to output results */

static void PrintOutput(realtype t, realtype y1, realtype y2, realtype y3);
static void PrintRootInfo(int root_f1, int root_f2);


/* Private function to print final statistics */

static void PrintFinalStats(void *cvode_mem);


/* Private function to check function return values */

static int check_flag(void *flagvalue, const char *funcname, int opt);



/* rate_equation_solver: solves the rate equations given in rate_equations.s                     */
/*-----------------------------------------------------------------------------------------------*/

int rate_equation_solver(long gridp)
{

  long *user_data;
  user_data = &gridp;


  int i;                                                                                /* index */

  realtype reltol, t, tout;
  N_Vector y, abstol;
  void *cvode_mem;
  int flag, flagr, iout;
  int rootsfound[2];

  y = abstol = NULL;
  cvode_mem = NULL;

  /* Create serial vector of length NEQ for I.C. and abstol */
  y = N_VNew_Serial(NEQ);
  if (check_flag((void *)y, "N_VNew_Serial", 0)) return(1);
  abstol = N_VNew_Serial(NEQ);
  if (check_flag((void *)abstol, "N_VNew_Serial", 0)) return(1);


  /* Initialize y */

  for (i=0; i<NSPEC; i++){

    Ith(y,i+1) = species[i].abn[gridp];
  }

  // Ith(y,1) = Y1;
  // Ith(y,2) = Y2;
  // Ith(y,3) = Y3;


  /* Set the scalar relative tolerance */
  reltol = RTOL;


  /* Set the vector absolute tolerance */

  for (i=0; i<NSPEC; i++){

    Ith(abstol,i+1) = ATOL;
  }

  // Ith(abstol,1) = ATOL1;
  // Ith(abstol,2) = ATOL2;
  // Ith(abstol,3) = ATOL3;



  /* Call CVodeCreate to create the solver memory and specify the
   * Backward Differentiation Formula and the use of a Newton iteration */
  cvode_mem = CVodeCreate(CV_BDF, CV_NEWTON);
  if (check_flag((void *)cvode_mem, "CVodeCreate", 0)) return(1);

  /* Call CVodeInit to initialize the integrator memory and specify the
   * user's right hand side function in y'=f(t,y), the inital time T0, and
   * the initial dependent variable vector y. */
  flag = CVodeInit(cvode_mem, f, T0, y);
  if (check_flag(&flag, "CVodeInit", 1)) return(1);

  /* Call CVodeSVtolerances to specify the scalar relative tolerance
   * and vector absolute tolerances */
  flag = CVodeSVtolerances(cvode_mem, reltol, abstol);
  if (check_flag(&flag, "CVodeSVtolerances", 1)) return(1);

  /* Call CVDense to specify the CVDENSE dense linear solver */
  flag = CVDense(cvode_mem, NEQ);
  if (check_flag(&flag, "CVDense", 1)) return(1);

  /* Set the Jacobian routine to Jac (user-supplied) */
  flag = CVDlsSetDenseJacFn(cvode_mem, Jac);
  if (check_flag(&flag, "CVDlsSetDenseJacFn", 1)) return(1);

  /* In loop, call CVode, print results, and test for error.
     Break out of loop when NOUT preset output times have been reached.  */
  printf(" \n3-species kinetics problem\n\n");


  iout = 0;  tout = T1;

  while(1) {

    flag = CVode(cvode_mem, tout, y, &t, CV_NORMAL);
    PrintOutput(t, Ith(y,1), Ith(y,2), Ith(y,3));

    if (check_flag(&flag, "CVode", 1)) break;
    if (flag == CV_SUCCESS) {
      iout++;
      tout *= TMULT;
    }

    if (iout == NOUT) break;

  }

  for (i=0; i<NSPEC; i++){

    species[i].abn[gridp] = Ith(y,i+1);
  }


  /* Print some final statistics */
  PrintFinalStats(cvode_mem);

  /* Free y and abstol vectors */
  N_VDestroy_Serial(y);
  N_VDestroy_Serial(abstol);

  /* Free integrator memory */
  CVodeFree(&cvode_mem);

  return(0);
}





/*   Private helper functions                                                                    */
/*_______________________________________________________________________________________________*/




/* PrintOutput                                                                                   */
/*-----------------------------------------------------------------------------------------------*/

static void PrintOutput(realtype t, realtype y1, realtype y2, realtype y3)
{
#if defined(SUNDIALS_EXTENDED_PRECISION)
  printf("At t = %0.4Le      y =%14.6Le  %14.6Le  %14.6Le\n", t, y1, y2, y3);
#elif defined(SUNDIALS_DOUBLE_PRECISION)
  printf("At t = %0.4e      y =%14.6e  %14.6e  %14.6e\n", t, y1, y2, y3);
#else
  printf("At t = %0.4e      y =%14.6e  %14.6e  %14.6e\n", t, y1, y2, y3);
#endif

  return;
}

/*-----------------------------------------------------------------------------------------------*/





/* PrintFinalStats: Get and print some final statistics                                          */
/*-----------------------------------------------------------------------------------------------*/

static void PrintFinalStats(void *cvode_mem)
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

/*-----------------------------------------------------------------------------------------------*/





/* check_flag                                                                                    */
/*-----------------------------------------------------------------------------------------------*/

static int check_flag(void *flagvalue, const char *funcname, int opt)
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

  /* Check if SUNDIALS function returned NULL pointer - no memory allocated */
  if (opt == 0 && flagvalue == NULL) {
    fprintf(stderr, "\nSUNDIALS_ERROR: %s() failed - returned NULL pointer\n\n",
	    funcname);
    return(1); }

  /* Check if flag < 0 */
  else if (opt == 1) {
    errflag = (int *) flagvalue;
    if (*errflag < 0) {
      fprintf(stderr, "\nSUNDIALS_ERROR: %s() failed with flag = %d\n\n",
	      funcname, *errflag);
      return(1); }}

  /* Check if function returned NULL pointer - no memory allocated */
  else if (opt == 2 && flagvalue == NULL) {
    fprintf(stderr, "\nMEMORY_ERROR: %s() failed - returned NULL pointer\n\n",
	    funcname);
    return(1); }

  return(0);
}

/*-----------------------------------------------------------------------------------------------*/
