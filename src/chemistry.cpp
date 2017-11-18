/* Frederik De Ceuster - University College London & KU Leuven                                   */
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

#include "../parameters.hpp"
#include "Magritte_config.hpp"
#include "declarations.hpp"

#include "chemistry.hpp"
#include "calc_column_density.hpp"
#include "reaction_rates.hpp"
#include "sundials/rate_equation_solver.hpp"
#include "write_output.hpp"



/* abundances: calculate abundances for each species at each grid point                          */
/*-----------------------------------------------------------------------------------------------*/


#ifdef ON_THE_FLY

int chemistry( GRIDPOINT *gridpoint,
               double *temperature_gas, double *temperature_dust, double *rad_surface, double *AV,
               double *column_H2, double *column_HD, double *column_C, double *column_CO )

#else

int chemistry( GRIDPOINT *gridpoint, EVALPOINT *evalpoint,
               long *key, long *raytot, long *cum_raytot,
               double *temperature_gas, double *temperature_dust, double *rad_surface, double *AV,
               double *column_H2, double *column_HD, double *column_C, double *column_CO )

#endif


{


  /* Calculate column densities */

# ifdef ON_THE_FLY

  calc_column_densities(gridpoint, column_H2, column_HD, column_C, column_CO);

# else

  calc_column_density(gridpoint, evalpoint, key, raytot, cum_raytot, column_H2, H2_nr);
  calc_column_density(gridpoint, evalpoint, key, raytot, cum_raytot, column_HD, HD_nr);
  calc_column_density(gridpoint, evalpoint, key, raytot, cum_raytot, column_C, C_nr);
  calc_column_density(gridpoint, evalpoint, key, raytot, cum_raytot, column_CO, CO_nr);

# endif


  /* For all gridpoints */

  for (long gridp=0; gridp<NGRID; gridp++){


    /* Calculate the reaction rates */

    reaction_rates( temperature_gas, temperature_dust, rad_surface, AV,
                    column_H2, column_HD, column_C, column_CO, gridp );


    /* Solve the rate equations */

    rate_equation_solver(gridpoint, gridp);


  } /* end of gridp loop over grid points */


  return(0);

}

/*-----------------------------------------------------------------------------------------------*/
