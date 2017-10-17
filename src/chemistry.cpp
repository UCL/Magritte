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

#include "declarations.hpp"
#include "chemistry.hpp"
#include "reaction_rates.hpp"
#include "sundials/rate_equation_solver.hpp"
#include "write_output.hpp"



/* abundances: calculate abundances for each species at each grid point                          */
/*-----------------------------------------------------------------------------------------------*/

int chemistry( GRIDPOINT *gridpoint, double *temperature_gas, double *temperature_dust,
               double *rad_surface, double *AV,
               double *column_H2, double *column_HD, double *column_C, double *column_CO,
               double v_turb )
{


  /* Output related variables, for testing only */

  int nr_can_reac = 0;
  int nr_can_phot = 0;
  int nr_all = 0;


  int canonical_reactions[NREAC];
  int can_photo_reactions[NREAC];
  int all_reactions[NREAC];

  /* ------------------------------------------ */


  /* For all gridpoints */

  for (long gridp=0; gridp<NGRID; gridp++){



    nr_can_reac = 0;
    nr_can_phot = 0;
    nr_all = 0;


    /* Calculate the reaction rates */

    reaction_rates( temperature_gas, temperature_dust, rad_surface, AV,
                    column_H2, column_HD, column_C, column_CO, v_turb, gridp,
                    &nr_can_reac, canonical_reactions,
                    &nr_can_phot, can_photo_reactions,
                    &nr_all, all_reactions );


    /* Solve the rate equations */

    rate_equation_solver(gridpoint, gridp);


  } /* end of gridp loop over grid points */



  /* Output related variables, for testing only */

  // write_certain_rates("0", "canonical", nr_can_reac, canonical_reactions, reaction);
  // write_certain_rates("0", "can_phot", nr_can_phot, can_photo_reactions, reaction);
  // write_certain_rates("0", "all", nr_all, all_reactions, reaction);
  //
  // write_double_1("rates2", "0", NGRID, &reaction[2].k[0]);

  /* ------------------------------------------ */



  return(0);

}

/*-----------------------------------------------------------------------------------------------*/
