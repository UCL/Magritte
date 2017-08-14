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
#include <stdlib.h>


/* Header files with a description of contents used */

#include <cvode/cvode.h>                                  /* prototypes for CVODE fcts., consts. */
#include <nvector/nvector_serial.h>                      /* serial N_Vector types, fcts., macros */
#include <cvode/cvode_dense.h>                                          /* prototype for CVDense */
#include <sundials/sundials_dense.h>                            /* definitions DlsMat DENSE_ELEM */
#include <sundials/sundials_types.h>                              /* definition of type realtype */

#include "../declarations.hpp"
#include "rate_equation_solver.hpp"
#include "jacobian.cpp"
#include "rate_equations.cpp"


/* User-defined vector and matrix accessor macros: Ith, IJth */

#define Ith(v,i)    NV_Ith_S(v,i)                             /* Ith numbers components 0..NEQ-1 */
#define IJth(A,i,j) DENSE_ELEM(A,i,j)                         /* IJth numbers rows,cols 0..NEQ-1 */


/* Problem Constants */

#define NEQ      (NSPEC-2)             /* number of equations: NSPEC minus dummy minus electrons */
#define RTOL     RCONST(1.0E-8)                                     /* scalar relative tolerance */
#define ATOL     RCONST(1.0e-30)                         /* vector absolute tolerance components */
#define T0       RCONST(0.0)                                                     /* initial time */
#define T1       RCONST(0.1)                                                /* first output time */
#define TMULT    RCONST(10.0)                                              /* output time factor */
#define NOUT_MAX 25                                                    /* number of output times */

#define OINDEX(outp,spe) ( (spe) + (NEQ+1)*(outp) )



/* rate_equation_solver: solves the rate equations given in rate_equations.s                     */
/*-----------------------------------------------------------------------------------------------*/

int rate_equation_solver(GRIDPOINT *gridpoint, long gridp)
{


  USER_DATA user_data;                               /* Data to be passed to the solver routines */

  user_data = NULL;
  user_data = (USER_DATA) malloc( sizeof(*user_data) );

  user_data->gp = gridp;
  user_data->gridpointer = gridpoint;


  int i;                                                                                /* index */

  realtype reltol, t, tout;
  N_Vector y, abstol;
  void *cvode_mem;
  int flag, flagr, nout, iout;
  int rootsfound[2];

  realtype seconds_in_year = RCONST(3.1556926e7);               /* Convert from years to seconds */

  realtype time_start = 0.0E0;                               /* start time of chemical evolution */

  realtype time_end = 1.0E7*seconds_in_year;                   /* end time of chemical evolution */

  y         = NULL;
  abstol    = NULL;
  cvode_mem = NULL;


  /* Specify the maximum number of internal steps */

  int mxstep = 10000000;






  /* Create serial vector of length NEQ for I.C. and abstol */

  y = N_VNew_Serial(NEQ);


  if (check_flag((void *)y, "N_VNew_Serial", 0)){

    return(1);
  }


  abstol = N_VNew_Serial(NEQ);

  if (check_flag((void *)abstol, "N_VNew_Serial", 0)){

    return(1);
  }



  /* Initialize y */

  for (i=0; i<NEQ; i++){

    Ith(y,i) = species[i+1].abn[gridp];
  }


  /* Set the scalar relative tolerance */

  reltol = RTOL;


  /* Set the vector absolute tolerance */

  for (i=0; i<NEQ; i++){

    Ith(abstol,i) = ATOL;
  }



  /* Call CVodeCreate to create the solver memory and specify the
   * Backward Differentiation Formula and the use of a Newton iteration */

  cvode_mem = CVodeCreate(CV_BDF, CV_NEWTON);

  if (check_flag((void *)cvode_mem, "CVodeCreate", 0)){

    return(1);
  }


  /* Call CVodeInit to initialize the integrator memory and specify the
   * user's right hand side function in y'=f(t,y), the inital time T0, and
   * the initial dependent variable vector y. */

  flag = CVodeInit(cvode_mem, f, T0, y);

  if (check_flag(&flag, "CVodeInit", 1)){

    return(1);
  }


  /* Call CVodeSVtolerances to specify the scalar relative tolerance
   * and vector absolute tolerances */

  flag = CVodeSVtolerances(cvode_mem, reltol, abstol);

  if (check_flag(&flag, "CVodeSVtolerances", 1)){

    return(1);
  }


  /* Call CVodeSetMaxNumSteps to set the maximum number of steps */

  flag = CVodeSetMaxNumSteps(cvode_mem, mxstep);

  if (check_flag(&flag, "CVodeSetMaxNumSteps", 1)){

    return(1);
  }


  /* Specify the user-defined data to be passed to the various routines */

  flag = CVodeSetUserData(cvode_mem, user_data);

  if (check_flag(&flag, "CVodeSetUserData", 1)){

    return(1);
  }


  /* Call CVDense to specify the CVDENSE dense linear solver */

  flag = CVDense(cvode_mem, NEQ);

  if (check_flag(&flag, "CVDense", 1)){

    return(1);
  }


  /* Set the Jacobian routine to Jac (user-supplied) */

  // flag = CVDlsSetDenseJacFn(cvode_mem, Jac);
  //
  // if (check_flag(&flag, "CVDlsSetDenseJacFn", 1)){
  //
  //   return(1);
  // }


  /* In loop, call CVode, print results, and test for error.
     Break out of loop when output time has been reached. */

  nout = NOUT_MAX;
  iout = 0;
  tout = 1.0E-4*seconds_in_year;

  realtype results[NOUT_MAX*(NEQ+1)];                        /* storage for intermediate results */

  /* Initialize */

  for(int t=0; t<NOUT_MAX*(NEQ+1); t++){

    results[t] = 0.0;
  }



  /* While the end time is not yet reached and there are no errors */

  while(1) {


    /* Call CVode, check the return status and loop */

    for (i=0; i<NEQ; i++){

      cout << "ic : " << Ith(y,i) << "\n";
    }


    /* Call CVode */

    flag = CVode(cvode_mem, tout, y, &t, CV_NORMAL);


    /* Store intermediate results */

    results[OINDEX(iout,0)] = t;

    for (int spe=0; spe<NEQ; spe++){

      results[OINDEX(iout,spe+1)] = Ith(y,spe);
    }


    if (check_flag(&flag, "CVode", 1)){

      printf("\n\n !!! CVode ERROR !!! \n\n");

      break;
    }


    if (flag == CV_SUCCESS) {

      iout++;

      tout = 10*tout;
    }


    if (tout > time_end || iout >= NOUT_MAX){

      nout = iout;

      break;
    }

  } /* End of while loop */



  /* Write the results of the integration */

  FILE *abn_file = fopen("output/abundances_in_time.txt", "w");

  if (abn_file == NULL){

      printf("Error opening file!\n");
      exit(1);
  }


  for (int outp=0; outp<NOUT_MAX; outp++){

    for (int spe=0; spe<NEQ+1; spe++){

      fprintf( abn_file, "%lE\t", results[OINDEX(outp,spe)] );
    }

    fprintf( abn_file, "\n" );
  }

  fclose(abn_file);



  /* Update the abundances for each species */

  for (i=0; i<NEQ; i++){

    species[i+1].abn[gridp] = Ith(y,i);
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

    fprintf( stderr, "\nSUNDIALS_ERROR: %s() failed - returned NULL pointer\n\n",
	           funcname );

    return(1);
  }


  /* Check if flag < 0 */

  else if (opt == 1) {

    errflag = (int *) flagvalue;

    if (*errflag < 0) {

      fprintf( stderr, "\nSUNDIALS_ERROR: %s() failed with flag = %d\n\n",
	             funcname, *errflag );

      return(1);
    }
  }


  /* Check if function returned NULL pointer - no memory allocated */

  else if (opt == 2 && flagvalue == NULL) {

    fprintf( stderr, "\nMEMORY_ERROR: %s() failed - returned NULL pointer\n\n",
	           funcname );

    return(1);
  }

  return(0);
}

/*-----------------------------------------------------------------------------------------------*/
