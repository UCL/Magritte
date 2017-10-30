/* Frederik De Ceuster - University College London & KU Leuven                                   */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/* update_temperatures_gas.c: Update the gas temperatures after a thermal balance iteration      */
/*                                                                                               */
/* (based on 3DPDR in 3DPDR)                                                                     */
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/
/*                                                                                               */
/*-----------------------------------------------------------------------------------------------*/



#include "../parameters.hpp"
#include "Magritte_config.hpp"
#include "declarations.hpp"

#include "update_temperature_gas.hpp"



/* update_temperature_gas: update the gas temperature after a thermal balance iteration          */
/*-----------------------------------------------------------------------------------------------*/

int update_temperature_gas( double thermal_flux, long gridp, double *temperature_gas,
                            double* previous_temperature_gas )
{


  /* When there is net heating, the temperature was too low -> increase temperature */

  if (thermal_flux > 0.0){


    /* When we also increrased the tempoerature previous iteration */

    if (previous_temperature_gas[gridp] < temperature_gas[gridp]){

      previous_temperature_gas[gridp] = temperature_gas[gridp];

      temperature_gas[gridp]          = 1.3 * temperature_gas[gridp];
    }


    /* When we decreased the temperature previous iteration */

    else {

      double temp = temperature_gas[gridp];

      temperature_gas[gridp]          = ( temp + previous_temperature_gas[gridp] ) / 2.0;

      previous_temperature_gas[gridp] = temp;
    }

  } /* end of net heating */



  /* When there is net cooling, the temperature was too high -> decrease temperature */

  if (thermal_flux < 0.0){


    /* When we also decrerased the tempoerature previous iteration */

    if (previous_temperature_gas[gridp] > temperature_gas[gridp]){

      previous_temperature_gas[gridp] = temperature_gas[gridp];

      temperature_gas[gridp]          = 0.7 * temperature_gas[gridp];
    }


    /* When we increased the temperature previous iteration */

    else {

      double temp = temperature_gas[gridp];

      temperature_gas[gridp]          = ( temp + previous_temperature_gas[gridp] ) / 2.0;

      previous_temperature_gas[gridp] = temp;
    }

  } /* end of net cooling */



  /* Enforce the minimun temperature to be T_CMB and maximum 30000 */

  if(temperature_gas[gridp] < T_CMB){

    temperature_gas[gridp] = T_CMB;
  }

  else if(temperature_gas[gridp] > 30000.0){

    temperature_gas[gridp] = 30000.0;
  }


  return(0);

}

/*-----------------------------------------------------------------------------------------------*/
