/* Frederik De Ceuster - University College London & KU Leuven                                   */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/* line_profile: Calculates the line profile function                                            */
/*                                                                                               */
/* (NEW)                                                                                         */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/



#include <math.h>

#include "../parameters.hpp"
#include "Magritte_config.hpp"
#include "declarations.hpp"

#include "line_profile.hpp"



/* line_profile: calculate the line profile function                                             */
/*-----------------------------------------------------------------------------------------------*/

double line_profile( EVALPOINT evalpoint, double *temperature_gas, double frequency,
                     long gridp, long evalp )
{

  double profile = 0.0;

  double velocity = evalpoint[GINDEX(gridp, GP_NR_OF_EVALP(gridp, evalp))].vol;

  double frequency_shift = frequency * (1.0 -  velocity/CC);


  return profile;

}

/*-----------------------------------------------------------------------------------------------*/
