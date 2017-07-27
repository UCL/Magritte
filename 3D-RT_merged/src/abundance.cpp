/* Frederik De Ceuster - University College London                                               */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/* abundance: Calculate abundances for each species at each grid point                           */
/*                                                                                               */
/* Calculate the abundances of all species at the specified end time based on their initial      */
/* abundances and the rates for each reaction. This routine calls the CVODE package to solve     */
/* for the set of ODEs. CVODE is able to handle stiff problems, where the dynamic range of the   */
/* rates can be very large.                                                                      */
/*                                                                                               */
/* (based on calculate_abundances in 3D-PDR)                                                     */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/





#include <stdio.h>
#include <stdlib.h>
// #include <omp.h>

#include "sundials/rate_equation_solver.c"


/* abundance: calculate abundances for each species at each grid point                           */
/*-----------------------------------------------------------------------------------------------*/

void abundance()
{

  long gridp;

  printf(" --- Abundance --- \n");

  int rate_equation_solver(long gridp);

  rate_equation_solver(gridp);

  printf(" ---           --- \n");

}

/*-----------------------------------------------------------------------------------------------*/
